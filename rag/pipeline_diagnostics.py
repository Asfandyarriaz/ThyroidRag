# rag/pipeline_diagnostics.py
"""
Complete Pipeline Diagnostics — JSON output, full pipeline
===========================================================
Traces every step of the QAPipeline and saves a single structured JSON report.

Steps traced:
  1  — Input question
  2  — Classification (which layer fired, raw LLM output, safety net details)
  3  — Query expansion (LLM sub-queries + domain fallbacks added)
  4  — Retrieval per sub-query (bi-encoder scores, titles, full text previews)
  5  — Deduplication (what was removed and why)
  6  — Cross-encoder reranking (before/after scores, rank changes)
  7  — Context construction (source map, char counts, truncation flag)
  8  — LLM prompt (full prompt sent, char count)
  9  — Raw LLM response (exact text, parse success/failure)
  10 — Parsed JSON answer (sections populated vs null, placeholder violations)
  11 — Faithfulness evaluation (per statement, per source, per score)
  12 — Final output summary (issues detected, recommendations)

Usage:
    from rag.pipeline_diagnostics import run_diagnostic

    report = run_diagnostic(
        pipeline=my_pipeline,
        question="What are common patterns of recurrence...",
        save_path="diagnosis.json"
    )

The returned dict is also saved to save_path as formatted JSON.
"""

import re
import json
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# EVIDENCE LEVEL WEIGHTS — mirrors qa_pipeline.py
# =============================================================================

EVIDENCE_LEVEL_WEIGHTS: Dict[int, Tuple[str, float]] = {
    1: ("Guidelines / Consensus", 1.00),
    2: ("Systematic Review / Meta-analysis", 0.90),
    3: ("Randomized Controlled Trials", 0.80),
    4: ("Clinical Trials (non-randomized)", 0.70),
    5: ("Cohort Studies", 0.60),
    6: ("Case-Control Studies", 0.50),
    7: ("Case Reports / Series", 0.40),
}

FIRST_STAGE_RETRIEVAL = 100
SECOND_STAGE_TOP_K = 20


# =============================================================================
# DIAGNOSTIC PIPELINE
# =============================================================================

class DiagnosticPipeline:
    """
    Wraps an existing QAPipeline and re-runs each step with full tracing.
    Does NOT modify the original pipeline object in any way.
    All output is written to a JSON file.
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, question: str, k: int = 30) -> Dict[str, Any]:
        """
        Run the full pipeline with complete tracing.

        Args:
            question : The user question to diagnose
            k        : Chunks to retrieve per sub-query (mirrors answer() default)

        Returns:
            Full diagnostic report as a nested dict
        """
        report: Dict[str, Any] = {
            "meta": {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "k_per_subquery": k,
            }
        }

        # ── Step 1: Input ─────────────────────────────────────────────────────
        report["step_1_input"] = {
            "question": question,
            "char_count": len(question),
            "word_count": len(question.split()),
        }

        # ── Step 2: Classification ────────────────────────────────────────────
        report["step_2_classification"] = self._trace_classification(question)
        question_type: str = report["step_2_classification"]["final_type"]

        # ── Step 3: Query expansion ───────────────────────────────────────────
        report["step_3_query_expansion"] = self._trace_query_expansion(question)
        sub_queries: List[str] = report["step_3_query_expansion"]["sub_queries"]

        # ── Step 4: Retrieval ─────────────────────────────────────────────────
        report["step_4_retrieval"], all_retrieved = self._trace_retrieval(
            sub_queries, k
        )

        # ── Step 5: Deduplication ─────────────────────────────────────────────
        report["step_5_deduplication"], unique_retrieved = self._trace_deduplication(
            all_retrieved
        )

        if not unique_retrieved:
            report["error"] = "No relevant information found after deduplication"
            report["step_12_final_output"] = {
                "error": "Pipeline halted — no chunks survived deduplication"
            }
            return report

        # ── Step 6: Cross-encoder reranking ───────────────────────────────────
        report["step_6_reranking"], reranked_chunks = self._trace_reranking(
            question, unique_retrieved
        )

        # ── Step 7: Context construction ──────────────────────────────────────
        report["step_7_context"], context, source_map = self._trace_context(
            reranked_chunks
        )

        # ── Steps 8 + 9 + 10: LLM prompt → raw response → parsed JSON ────────
        llm_result = self._trace_llm(question, context, question_type)
        report["step_8_llm_prompt"]      = llm_result["prompt_trace"]
        report["step_9_raw_response"]    = llm_result["raw_response_trace"]
        report["step_10_parsed_answer"]  = llm_result["parsed_answer_trace"]
        json_response                    = llm_result.get("json_response")

        # ── Step 11: Faithfulness ─────────────────────────────────────────────
        if json_response:
            report["step_11_faithfulness"] = self._trace_faithfulness(
                json_response, context, source_map
            )
        else:
            report["step_11_faithfulness"] = {
                "skipped": True,
                "reason": "No valid JSON response — JSON parse failed in step 9"
            }

        # ── Step 12: Final summary + issue detection ───────────────────────────
        report["step_12_final_output"] = self._build_summary(report)

        return report

    def save(self, report: Dict[str, Any], filepath: str) -> None:
        """Save diagnostic report as formatted JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Diagnostic report saved → {filepath}")
        print(f"\n✅ Diagnostic report saved → {filepath}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP TRACERS
    # ─────────────────────────────────────────────────────────────────────────

    def _trace_classification(self, question: str) -> Dict[str, Any]:
        """
        Re-run all three classification layers and record exactly what happened.
        """
        p = self.pipeline
        result: Dict[str, Any] = {
            "question": question,
            "layer_1_keyword_result": None,
            "layer_2_llm_raw_output": None,
            "layer_2_llm_validated_category": None,
            "layer_2_llm_fallback_triggered": False,
            "layer_2_llm_error": None,
            "layer_3_diagnostic_tool_applied": False,
            "layer_3_evidence_applied": False,
            "layer_3_molecular_applied": False,
            "layer_3_detail": [],
            "final_type": None,
            "layer_fired": None,
        }

        # Layer 1 — keyword pre-check
        kw = p._keyword_preclassify(question)
        result["layer_1_keyword_result"] = kw
        if kw:
            result["final_type"] = kw
            result["layer_fired"] = "Layer 1 — keyword pre-check (deterministic)"
            return result

        # Layer 2 — LLM classifier
        valid_categories = [
            "definition", "complications", "comparison", "treatment",
            "diagnosis", "timing", "evidence", "staging",
            "risk_stratification", "impact", "surveillance",
            "recurrence", "molecular"
        ]
        classification_prompt = f"""Classify this thyroid cancer question into ONE category.

Categories: definition, complications, comparison, treatment, diagnosis, timing,
evidence, staging, risk_stratification, impact, surveillance, recurrence, molecular

Question: {question}

Return ONLY the category name (one word), nothing else:"""

        category = "definition"
        try:
            raw = p.llm.ask(classification_prompt).strip().lower()
            result["layer_2_llm_raw_output"] = raw
            if raw in valid_categories:
                category = raw
                result["layer_2_llm_validated_category"] = raw
            else:
                result["layer_2_llm_fallback_triggered"] = True
                result["layer_2_llm_validated_category"] = "definition (fallback)"
                result["layer_2_detail"] = f"Raw output '{raw}' not in valid_categories → defaulted to 'definition'"
        except Exception as e:
            result["layer_2_llm_error"] = str(e)

        # Layer 3 — safety nets
        original = category

        if category == "definition":
            after = p._reclassify_if_diagnostic_tool(question, category)
            if after != category:
                result["layer_3_diagnostic_tool_applied"] = True
                result["layer_3_detail"].append(
                    f"_reclassify_if_diagnostic_tool: '{original}' → '{after}'"
                )
                category = after

        if category == "definition":
            after = p._reclassify_if_evidence_question(question, category)
            if after != category:
                result["layer_3_evidence_applied"] = True
                result["layer_3_detail"].append(
                    f"_reclassify_if_evidence_question: '{original}' → '{after}'"
                )
                category = after

        if category == "definition":
            after = p._reclassify_if_molecular_question(question, category)
            if after != category:
                result["layer_3_molecular_applied"] = True
                result["layer_3_detail"].append(
                    f"_reclassify_if_molecular_question: '{original}' → '{after}'"
                )
                category = after

        layer_3_fired = any([
            result["layer_3_diagnostic_tool_applied"],
            result["layer_3_evidence_applied"],
            result["layer_3_molecular_applied"],
        ])

        result["final_type"] = category
        result["layer_fired"] = (
            "Layer 3 — post-LLM safety net" if layer_3_fired
            else "Layer 2 — LLM classifier"
        )
        return result

    def _trace_query_expansion(self, question: str) -> Dict[str, Any]:
        """Re-run query expansion and capture LLM raw output + fallback additions."""
        p = self.pipeline
        result: Dict[str, Any] = {
            "question": question,
            "expansion_source": None,
            "llm_raw_output": None,
            "llm_parse_error": None,
            "llm_error": None,
            "queries_from_llm": [],
            "queries_added_by_domain_fallback": [],
            "sub_queries": [],
            "total_sub_queries": 0,
        }

        expansion_prompt = f"""You are a medical information retrieval assistant specialised in thyroid cancer.
Generate 3-5 targeted search queries to retrieve comprehensive information.
Include the original question. Use specific medical terminology.

Question: {question}

Return ONLY a JSON array of 3-5 search queries, no other text:"""

        try:
            raw = p.llm.ask(expansion_prompt)
            result["llm_raw_output"] = raw

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```\s*$', '', cleaned)

            queries = json.loads(cleaned)

            if isinstance(queries, list) and len(queries) > 0:
                if question not in queries:
                    queries.insert(0, question)

                llm_queries = list(queries)
                result["queries_from_llm"] = llm_queries

                # Apply domain fallbacks and track what was added
                expanded = p._add_fallback_queries(question, list(queries))
                added = [q for q in expanded if q not in llm_queries]
                result["queries_added_by_domain_fallback"] = added
                result["sub_queries"] = expanded
                result["expansion_source"] = "LLM expansion + domain fallbacks"
            else:
                fallback = p._create_fallback_queries(question)
                result["sub_queries"] = fallback
                result["expansion_source"] = "Rule-based fallback (LLM returned invalid list)"

        except json.JSONDecodeError as e:
            result["llm_parse_error"] = str(e)
            fallback = p._create_fallback_queries(question)
            result["sub_queries"] = fallback
            result["expansion_source"] = "Rule-based fallback (JSON parse error)"

        except Exception as e:
            result["llm_error"] = str(e)
            fallback = p._create_fallback_queries(question)
            result["sub_queries"] = fallback
            result["expansion_source"] = "Rule-based fallback (LLM exception)"

        result["total_sub_queries"] = len(result["sub_queries"])
        return result

    def _trace_retrieval(
        self, sub_queries: List[str], k: int
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """Retrieve chunks for each sub-query and record full results."""
        p = self.pipeline
        chunks_per_query = FIRST_STAGE_RETRIEVAL // len(sub_queries) if sub_queries else 10

        result: Dict[str, Any] = {
            "chunks_per_query": chunks_per_query,
            "total_chunks_retrieved": 0,
            "per_query_results": [],
        }
        all_retrieved: List[Dict] = []

        for idx, sub_query in enumerate(sub_queries, 1):
            chunks = p.vector_store.search(sub_query, k=chunks_per_query)
            all_retrieved.extend(chunks)

            # Capture ALL chunks for this query, not just top 3
            chunks_detail = []
            for rank, chunk in enumerate(chunks, 1):
                chunks_detail.append({
                    "rank": rank,
                    "bi_encoder_score": round(float(chunk.get("score", 0) or 0), 6),
                    "title": chunk.get("title", "No title"),
                    "pmid": chunk.get("pmid", "Unknown"),
                    "doi": chunk.get("doi", "Unknown"),
                    "year": chunk.get("year", "Unknown"),
                    "evidence_level": chunk.get("evidence_level", "Unknown"),
                    "text_length": len(chunk.get("text", "")),
                    "text_full": chunk.get("text", ""),          # full text
                    "text_preview_300": chunk.get("text", "")[:300],
                })

            result["per_query_results"].append({
                "query_index": idx,
                "query": sub_query,
                "chunks_returned": len(chunks),
                "chunks": chunks_detail,
                # Convenience: top 5 for quick inspection
                "top_5_preview": [
                    {
                        "rank": c["rank"],
                        "score": c["bi_encoder_score"],
                        "title": c["title"],
                        "pmid": c["pmid"],
                        "evidence_level": c["evidence_level"],
                        "text_preview": c["text_preview_300"],
                    }
                    for c in chunks_detail[:5]
                ],
            })

        result["total_chunks_retrieved"] = len(all_retrieved)
        return result, all_retrieved

    def _trace_deduplication(
        self, chunks: List[Dict]
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """Deduplicate and record what was removed."""
        seen: set = set()
        unique: List[Dict] = []
        removed: List[Dict] = []

        for chunk in chunks:
            text_snippet = chunk.get("text", "")[:200].strip()
            pmid = chunk.get("pmid", "unknown")
            chunk_id = f"{pmid}||{hash(text_snippet)}"
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk)
            else:
                removed.append(chunk)

        result: Dict[str, Any] = {
            "before": len(chunks),
            "after": len(unique),
            "removed_count": len(removed),
            "dedup_key_format": "PMID + hash of first 200 chars of text",
            "removed_chunks": [
                {
                    "pmid": c.get("pmid", "Unknown"),
                    "title": c.get("title", "No title"),
                    "text_snippet": c.get("text", "")[:150],
                }
                for c in removed
            ],
        }
        return result, unique

    def _trace_reranking(
        self, question: str, chunks: List[Dict]
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """Run cross-encoder reranking and record all score changes."""
        p = self.pipeline

        # Record original bi-encoder scores and ranks
        original_index: Dict[str, Dict] = {}
        for i, c in enumerate(chunks):
            key = f"{c.get('pmid','?')}||{hash(c.get('text','')[:200].strip())}"
            original_index[key] = {
                "bi_encoder_score": round(float(c.get("score", 0) or 0), 6),
                "original_rank": i + 1,
                "title": c.get("title", ""),
                "pmid": c.get("pmid", ""),
            }

        # Run reranking on a deep copy so original list is unchanged
        reranked = p._rerank_with_cross_encoder(
            question=question,
            chunks=deepcopy(chunks),
            top_k=SECOND_STAGE_TOP_K
        )

        bi_scores_all = [float(c.get("score", 0) or 0) for c in chunks]

        # Build per-chunk detail for all reranked chunks
        reranked_detail = []
        for new_rank, chunk in enumerate(reranked, 1):
            key = f"{chunk.get('pmid','?')}||{hash(chunk.get('text','')[:200].strip())}"
            orig = original_index.get(key, {})
            ce_score = round(float(chunk.get("score", 0)), 6)
            be_score = round(float(orig.get("bi_encoder_score", 0)), 6)
            orig_rank = orig.get("original_rank", "?")
            reranked_detail.append({
                "new_rank": new_rank,
                "original_rank": orig_rank,
                "rank_delta": (orig_rank - new_rank) if isinstance(orig_rank, int) else None,
                "cross_encoder_score": ce_score,
                "bi_encoder_score": be_score,
                "title": chunk.get("title", ""),
                "pmid": chunk.get("pmid", ""),
                "year": chunk.get("year", ""),
                "evidence_level": chunk.get("evidence_level", ""),
                "text_preview_300": chunk.get("text", "")[:300],
                "text_full": chunk.get("text", ""),
            })

        # Significant rank changes (moved ≥5 positions)
        rank_changes = [
            r for r in reranked_detail
            if r["rank_delta"] is not None and abs(r["rank_delta"]) >= 5
        ]
        rank_changes.sort(key=lambda x: abs(x["rank_delta"] or 0), reverse=True)

        ce_scores = [r["cross_encoder_score"] for r in reranked_detail]

        result: Dict[str, Any] = {
            "input_chunk_count": len(chunks),
            "output_chunk_count": len(reranked),
            "bi_encoder_score_min": round(min(bi_scores_all), 6) if bi_scores_all else None,
            "bi_encoder_score_max": round(max(bi_scores_all), 6) if bi_scores_all else None,
            "cross_encoder_score_min": round(min(ce_scores), 6) if ce_scores else None,
            "cross_encoder_score_max": round(max(ce_scores), 6) if ce_scores else None,
            "all_reranked_chunks": reranked_detail,
            "significant_rank_changes": rank_changes,
        }
        return result, reranked

    def _trace_context(
        self, reranked_chunks: List[Dict]
    ) -> Tuple[Dict[str, Any], str, Dict]:
        """Build context and record what made it in and what was cut."""
        p = self.pipeline
        context, source_map = p._build_tagged_context(reranked_chunks)

        # Annotate source map
        annotated_map: Dict[str, Any] = {}
        for tag, meta in source_map.items():
            lvl = meta.get("evidence_level")
            ev_desc, ev_weight = EVIDENCE_LEVEL_WEIGHTS.get(lvl, ("Unknown", 0.0)) if lvl else ("Unknown", 0.0)
            annotated_map[tag] = {
                **meta,
                "evidence_level_description": ev_desc,
                "evidence_weight": ev_weight,
            }

        result: Dict[str, Any] = {
            "source_count": len(source_map),
            "total_context_chars": len(context),
            "truncated_at_limit": len(context) >= 9800,
            "source_map": annotated_map,
            "full_tagged_context": context,        # complete context sent to LLM
            "context_preview_1000": context[:1000],
        }
        return result, context, source_map

    def _trace_llm(
        self, question: str, context: str, question_type: str
    ) -> Dict[str, Any]:
        """
        Generate the LLM prompt, call the LLM, capture raw response,
        and attempt JSON parsing.
        """
        p = self.pipeline
        prompt = p._create_type_specific_prompt(question, context, question_type)

        # ── Step 8: Prompt ────────────────────────────────────────────────────
        prompt_trace: Dict[str, Any] = {
            "question_type": question_type,
            "prompt_char_count": len(prompt),
            "prompt_word_count": len(prompt.split()),
            "full_prompt": prompt,                 # complete prompt
            "prompt_preview_1000": prompt[:1000],
            "prompt_tail_500": prompt[-500:],
        }

        # ── Step 9: Raw response ──────────────────────────────────────────────
        raw_response = ""
        try:
            raw_response = p.llm.ask(prompt).strip()
        except Exception as e:
            raw_response = f"LLM_CALL_ERROR: {e}"

        had_fence = raw_response.startswith("```")
        cleaned = raw_response
        if had_fence:
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)

        json_response = None
        json_parse_success = False
        json_parse_error = None
        try:
            json_response = json.loads(cleaned)
            json_parse_success = True
        except json.JSONDecodeError as e:
            json_parse_error = str(e)

        raw_response_trace: Dict[str, Any] = {
            "response_char_count": len(raw_response),
            "had_markdown_fence": had_fence,
            "json_parse_success": json_parse_success,
            "json_parse_error": json_parse_error,
            "full_raw_response": raw_response,     # complete raw text
            "raw_response_preview_1000": raw_response[:1000],
        }

        # ── Step 10: Parsed JSON analysis ─────────────────────────────────────
        parsed_trace: Dict[str, Any] = {
            "json_parse_success": json_parse_success,
        }

        if json_response:
            overview = json_response.get("overview", "")
            sections = json_response.get("sections", [])

            populated_sections: List[str] = []
            null_sections: List[str] = []
            placeholder_violations: List[str] = []
            section_details: List[Dict] = []

            PLACEHOLDERS = [
                "[Topic]", "[Drug Name]", "[X]", "Type Name",
                "Procedure Name", "Treatment Name", "Name here",
                "Insert here", "[Insert]"
            ]

            for sec in sections:
                header = sec.get("header", "Unknown section")
                sec_json = json.dumps(sec)

                # Check for content
                has_real_content = False
                for field in ["content", "items", "steps", "table", "subgroups", "criteria", "comparison_table"]:
                    val = sec.get(field)
                    if val and val != "null":
                        if isinstance(val, list) and len(val) > 0:
                            has_real_content = True
                        elif isinstance(val, str) and val.strip():
                            has_real_content = True

                if has_real_content:
                    populated_sections.append(header)
                else:
                    null_sections.append(header)

                # Check for placeholders
                for ph in PLACEHOLDERS:
                    if ph in sec_json:
                        placeholder_violations.append(
                            f"Section '{header}' contains placeholder text: '{ph}'"
                        )

                section_details.append({
                    "header": header,
                    "is_populated": has_real_content,
                    "fields_present": [
                        f for f in ["content", "items", "steps", "table",
                                    "subgroups", "criteria", "comparison_table"]
                        if sec.get(f) is not None
                    ],
                    "content_preview": sec_json[:400],
                })

            parsed_trace.update({
                "has_overview": bool(overview and overview != "null"),
                "overview_text": overview,
                "total_sections": len(sections),
                "populated_sections": populated_sections,
                "null_sections": null_sections,
                "placeholder_violations": placeholder_violations,
                "section_details": section_details,
                "full_json_response": json_response,  # complete parsed answer
            })

        return {
            "prompt_trace": prompt_trace,
            "raw_response_trace": raw_response_trace,
            "parsed_answer_trace": parsed_trace,
            "json_response": json_response,
        }

    def _trace_faithfulness(
        self, json_response: Dict, context: str, source_map: Dict
    ) -> Dict[str, Any]:
        """
        Re-run faithfulness evaluation with full per-statement, per-source detail.
        """
        p = self.pipeline
        ev = p.faithfulness_evaluator

        statements = ev._extract_statements(json_response)
        source_contexts = ev._extract_source_contexts(context)

        total_score = 0.0
        evaluated = 0
        skipped = 0
        faithful_count = 0
        partial_count = 0
        not_faithful_count = 0
        statement_details: List[Dict] = []

        for statement_text, cited_sources in statements:
            if not cited_sources:
                skipped += 1
                statement_details.append({
                    "statement": statement_text,
                    "cited_sources": [],
                    "score": None,
                    "verdict": "SKIPPED — no [SOURCE_X] citation",
                    "per_source_results": [],
                })
                continue

            per_source: List[Dict] = []
            scores: List[float] = []

            for source_id in cited_sources:
                if source_id not in source_contexts:
                    scores.append(0.0)
                    per_source.append({
                        "source_id": source_id,
                        "available_in_context": False,
                        "score": 0.0,
                        "label": "NOT_FAITHFUL",
                        "reason": "Source cited but no context chunk available",
                        "context_used": None,
                    })
                    continue

                ctx = source_contexts[source_id]
                score = ev._evaluate_statement(statement_text, ctx, source_id)
                label = (
                    "FAITHFUL" if score == 1.0
                    else "PARTIAL" if score == 0.5
                    else "NOT_FAITHFUL"
                )
                scores.append(score)
                per_source.append({
                    "source_id": source_id,
                    "available_in_context": True,
                    "score": score,
                    "label": label,
                    "context_preview_300": ctx[:300],
                })

            avg = sum(scores) / len(scores) if scores else 0.0
            total_score += avg
            evaluated += 1

            if avg == 1.0:
                faithful_count += 1
                verdict = "FAITHFUL"
            elif avg >= 0.5:
                partial_count += 1
                verdict = "PARTIAL"
            else:
                not_faithful_count += 1
                verdict = "NOT_FAITHFUL"

            statement_details.append({
                "statement": statement_text,
                "cited_sources": cited_sources,
                "score": round(avg, 4),
                "verdict": verdict,
                "per_source_results": per_source,
            })

        overall_score = total_score / evaluated if evaluated > 0 else 0.0
        label = (
            "High" if overall_score >= 0.80
            else "Medium" if overall_score >= 0.60
            else "Low"
        )

        return {
            "overall_score": round(overall_score, 4),
            "label": label,
            "total_statements_extracted": len(statements),
            "evaluated_statements": evaluated,
            "skipped_no_citation": skipped,
            "faithful_count": faithful_count,
            "partial_count": partial_count,
            "not_faithful_count": not_faithful_count,
            "faithfulness_thresholds": {
                "High": "≥ 0.80",
                "Medium": "≥ 0.60",
                "Low": "< 0.60"
            },
            "statement_details": statement_details,
        }

    def _build_summary(self, report: Dict) -> Dict[str, Any]:
        """
        Analyse all steps and produce:
        - key metrics
        - issues detected
        - specific recommendations
        """
        s2  = report.get("step_2_classification", {})
        s3  = report.get("step_3_query_expansion", {})
        s4  = report.get("step_4_retrieval", {})
        s5  = report.get("step_5_deduplication", {})
        s6  = report.get("step_6_reranking", {})
        s7  = report.get("step_7_context", {})
        s9  = report.get("step_9_raw_response", {})
        s10 = report.get("step_10_parsed_answer", {})
        s11 = report.get("step_11_faithfulness", {})

        # Evidence confidence
        src_map = s7.get("source_map", {})
        levels = [
            v.get("evidence_level") for v in src_map.values()
            if v.get("evidence_level") in EVIDENCE_LEVEL_WEIGHTS
        ]
        if levels:
            weights = [EVIDENCE_LEVEL_WEIGHTS[l][1] for l in levels]
            conf_score = int(round(sum(weights) / len(weights) * 100))
            conf_label = "High" if conf_score >= 85 else "Medium" if conf_score >= 65 else "Low"
            confidence = f"{conf_label} ({conf_score}%)"
        else:
            confidence = "Unknown — no evidence_level metadata in retrieved chunks"

        # ── Issue detection ───────────────────────────────────────────────────
        issues: List[str] = []
        recommendations: List[str] = []

        # Classification
        if s2.get("layer_fired") == "Layer 2 — LLM classifier":
            issues.append(
                "Classification relied on LLM (no keyword pre-check matched). "
                "Verify the final_type is correct — LLM classifiers can drift."
            )

        if s2.get("layer_2_llm_fallback_triggered"):
            issues.append(
                f"LLM returned an invalid category '{s2.get('layer_2_llm_raw_output')}' "
                f"and fell back to 'definition'. Check LLM classification prompt."
            )

        # Query expansion
        if s3.get("expansion_source", "").startswith("Rule-based"):
            issues.append(
                f"Query expansion fell back to rule-based ({s3.get('expansion_source')}). "
                f"LLM expansion failed — check LLM connectivity."
            )

        if len(s3.get("queries_added_by_domain_fallback", [])) == 0:
            issues.append(
                "No domain-specific fallback queries were added. "
                "This may result in narrow retrieval for specialised question types."
            )

        # Retrieval
        null_secs = s10.get("null_sections", [])
        populated_secs = s10.get("populated_sections", [])

        if null_secs:
            issues.append(
                f"{len(null_secs)} answer section(s) are null/empty: "
                f"{', '.join(null_secs)}. "
                f"The LLM found no relevant content in retrieved chunks for these topics."
            )
            recommendations.append(
                f"ROOT CAUSE OF EMPTY SECTIONS: Check step_4_retrieval.per_query_results "
                f"and look at chunk text_full fields. If chunks do not mention "
                f"{', '.join(null_secs)}, you need more targeted sub-queries for those topics. "
                f"Add specific phrases like '{null_secs[0].lower()} thyroid cancer' to _add_fallback_queries."
            )

        total_retrieved = s4.get("total_chunks_retrieved", 0)
        after_dedup = s5.get("after", 0)
        after_rerank = s6.get("output_chunk_count", 0)

        if total_retrieved < 10:
            issues.append(
                f"Very few chunks retrieved ({total_retrieved}). "
                f"Vector store may not contain documents relevant to this question."
            )
            recommendations.append(
                "Check if your Qdrant collection contains papers on this topic. "
                "Run a direct vector_store.search() with the raw question to confirm."
            )

        if after_rerank < 5:
            issues.append(
                f"Only {after_rerank} chunks survived to context. "
                f"Context may be too thin to answer the question fully."
            )

        # Context truncation
        if s7.get("truncated_at_limit"):
            issues.append(
                "Context was truncated at MAX_TOTAL_CONTEXT_CHARS (10000). "
                "Some retrieved content was not sent to the LLM."
            )
            recommendations.append(
                "Consider reducing MAX_EXCERPT_CHARS (currently 1200) or "
                "MAX_CHUNKS_PER_SOURCE (currently 3) to fit more diverse sources "
                "within the character limit."
            )

        # JSON parse
        if not s9.get("json_parse_success"):
            issues.append(
                f"JSON parse failed: {s9.get('json_parse_error')}. "
                f"The LLM did not return valid JSON. Check full_raw_response in step_9."
            )
            recommendations.append(
                "If LLM output contains markdown fences or prose before JSON, "
                "the regex cleanup in answer() should handle it. "
                "If it returns prose instead of JSON, the prompt's schema contract "
                "may need stronger instruction."
            )

        # Placeholder violations
        if s10.get("placeholder_violations"):
            for v in s10["placeholder_violations"]:
                issues.append(f"Schema violation — {v}")
            recommendations.append(
                "Placeholder text in output means the LLM is treating the JSON template "
                "as fill-in-the-blank rather than generating real content. "
                "Ensure the STRICT SCHEMA CONTRACT block is present in the prompt."
            )

        # Faithfulness
        if not s11.get("skipped"):
            faith_score = s11.get("overall_score", 1.0)
            if faith_score is not None and faith_score < 0.6:
                issues.append(
                    f"Low faithfulness score: {faith_score:.2f} ({s11.get('label')}). "
                    f"The LLM is generating content not well-supported by retrieved chunks."
                )
                recommendations.append(
                    "Low faithfulness usually means either (a) wrong chunks were retrieved "
                    "so the LLM had to speculate, or (b) the LLM ignored source tags. "
                    "Check step_11_faithfulness.statement_details for NOT_FAITHFUL statements "
                    "and compare against step_7_context.full_tagged_context."
                )

        # ── Final summary ─────────────────────────────────────────────────────
        return {
            "question": report["meta"]["question"],
            "timestamp": report["meta"]["timestamp"],

            "classification": {
                "final_type": s2.get("final_type"),
                "layer_fired": s2.get("layer_fired"),
            },

            "retrieval_stats": {
                "sub_queries_generated": s3.get("total_sub_queries", 0),
                "chunks_stage_1_bi_encoder": total_retrieved,
                "chunks_after_deduplication": after_dedup,
                "chunks_after_reranking": after_rerank,
                "sources_in_final_context": s7.get("source_count", 0),
                "context_chars": s7.get("total_context_chars", 0),
                "context_truncated": s7.get("truncated_at_limit", False),
            },

            "answer_quality": {
                "json_parse_success": s9.get("json_parse_success", False),
                "sections_populated": populated_secs,
                "sections_null": null_secs,
                "placeholder_violations": s10.get("placeholder_violations", []),
                "evidence_confidence": confidence,
                "faithfulness_score": s11.get("overall_score"),
                "faithfulness_label": s11.get("label"),
            },

            "issues_detected": issues,
            "recommendations": recommendations,
            "issue_count": len(issues),
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_diagnostic(
    pipeline: Any,
    question: str,
    save_path: str = "diagnostic_report.json",
    k: int = 30,
) -> Dict[str, Any]:
    """
    Run full pipeline diagnostic and save results to JSON.

    Args:
        pipeline  : Initialised QAPipeline instance
        question  : The question to diagnose
        save_path : File path to save the JSON report
        k         : Chunks per sub-query (should match your answer() call)

    Returns:
        The full diagnostic report dict
    """
    diag = DiagnosticPipeline(pipeline)
    logger.info(f"Starting diagnostic for: '{question}'")
    report = diag.run(question=question, k=k)
    diag.save(report, save_path)
    return report
