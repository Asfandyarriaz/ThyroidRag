# rag/qa_pipeline.py
import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import CrossEncoder

# Import faithfulness evaluator
try:
    from .faithfulness_evaluator import FaithfulnessEvaluator
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from faithfulness_evaluator import FaithfulnessEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evidence level definitions for confidence scoring
EVIDENCE_LEVEL_WEIGHTS: Dict[int, Tuple[str, float]] = {
    1: ("Guidelines / Consensus", 1.00),
    2: ("Systematic Review / Meta-analysis", 0.90),
    3: ("Randomized Controlled Trials", 0.80),
    4: ("Clinical Trials (non-randomized)", 0.70),
    5: ("Cohort Studies", 0.60),
    6: ("Case-Control Studies", 0.50),
    7: ("Case Reports / Series", 0.40),
}

# Retrieval configuration
FIRST_STAGE_RETRIEVAL = 100  # Bi-encoder retrieval (broad)
SECOND_STAGE_TOP_K = 20      # Cross-encoder re-ranking (precise)
MAX_SOURCES = 10
MAX_CHUNKS_PER_SOURCE = 3
MAX_EXCERPT_CHARS = 1200
MAX_TOTAL_CONTEXT_CHARS = 10000


class QAPipeline:
    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        llm_client: Any,
        instruction_file: str = "instructions/rag_instructions.txt",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm_client

        # Load instructions
        env = os.getenv("ENV", "local").lower()
        base = Path(__file__).parent if env == "prod" else Path(".")
        self.instructions = (base / instruction_file).read_text(encoding="utf-8")

        # Initialize cross-encoder for re-ranking
        logger.info(f"Loading cross-encoder model: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        logger.info("Cross-encoder loaded successfully")

        # Initialize faithfulness evaluator
        logger.info("Initializing faithfulness evaluator")
        self.faithfulness_evaluator = FaithfulnessEvaluator(self.llm)
        logger.info("Faithfulness evaluator initialized")

    # =========================================================================
    # STEP 1 — CLASSIFICATION (three-layer defence)
    # =========================================================================

    def _classify_question_type(self, question: str) -> str:
        """
        Classify the question type to use the appropriate answer template.

        Three-layer defence against misclassification:
          Layer 1 — keyword pre-check  (before LLM, deterministic)
          Layer 2 — LLM classifier     (handles ambiguous cases)
          Layer 3 — post-LLM safety nets (reclassify if LLM still wrong)
        """

        # ── LAYER 1: keyword pre-check ────────────────────────────────────────
        # Runs BEFORE the LLM. Returns immediately for obvious patterns.
        # Prevents LLM from misreading "What is the evidence for..." as definition.
        pre_check = self._keyword_preclassify(question)
        if pre_check:
            logger.info(f"Question pre-classified by keyword as: {pre_check}")
            return pre_check

        # ── LAYER 2: LLM classifier ───────────────────────────────────────────
        classification_prompt = f"""Classify this thyroid cancer question into ONE category.

Categories:
- definition:    "What is X?", "Tell me about X", "Explain X", "Describe X"
- complications: "What are complications/risks/side effects of X?", "What risks?"
- comparison:    "X vs Y", "difference between X and Y", "compare X and Y"
- treatment:     "How to treat X?", "What are treatment options?", "How is X treated?"
- diagnosis:     "How is X diagnosed?", "What tests for X?", "How to detect X?"
- timing:        "When should X?", "When is X recommended?", "When to do X?"
- evidence:      "What is the evidence for X?", "What does research/trials show?",
                 "What data exists for X?", "Is X effective?", "Efficacy of X?",
                 "What are outcomes of X?", "What studies support X?"

Question: {question}

IMPORTANT: If the question contains "evidence", "efficacy", "trial", "outcomes",
"research shows", or "studies show" — classify as "evidence", not "definition".

Return ONLY the category name (one word), nothing else:"""

        try:
            category = self.llm.ask(classification_prompt).strip().lower()

            valid_categories = [
                "definition", "complications", "comparison",
                "treatment", "diagnosis", "timing", "evidence"
            ]
            if category not in valid_categories:
                logger.warning(f"Invalid category '{category}', defaulting to 'definition'")
                category = "definition"

            # ── LAYER 3: post-LLM safety nets ─────────────────────────────────
            # Only run if LLM returned definition — the most commonly wrong label
            if category == "definition":
                category = self._reclassify_if_diagnostic_tool(question, category)
            if category == "definition":
                category = self._reclassify_if_evidence_question(question, category)

            logger.info(f"Question classified as: {category}")
            return category

        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            return "definition"

    def _keyword_preclassify(self, question: str) -> Optional[str]:
        """
        Fast keyword-based pre-classification that bypasses the LLM entirely.

        Returns a category string if confident, or None to fall through to LLM.
        Catches patterns the LLM consistently gets wrong — particularly evidence
        questions that start with "What is..." being labelled as definition.
        """
        q = question.lower().strip()

        # ── Evidence ──────────────────────────────────────────────────────────
        evidence_trigger_phrases = [
            "what is the evidence",
            "what is the clinical evidence",
            "what evidence exists",
            "what does the evidence",
            "what does research show",
            "what does the research show",
            "what do studies show",
            "what do the studies show",
            "what clinical trials",
            "what trials exist",
            "what trials support",
            "evidence for",
            "evidence of",
            "evidence supporting",
            "evidence behind",
            "evidence base for",
            "evidence-based",
            "is there evidence",
            "what are the outcomes of",
            "what are clinical outcomes",
            "efficacy of",
            "how effective is",
            "how efficacious",
            "what is the efficacy",
            "clinical evidence for",
            "trial data",
            "trial evidence",
        ]
        if any(phrase in q for phrase in evidence_trigger_phrases):
            return "evidence"

        # ── Complications ─────────────────────────────────────────────────────
        complication_trigger_phrases = [
            "what are the complications",
            "what are complications",
            "what are the risks of",
            "what are risks of",
            "what are the side effects",
            "what are side effects",
            "what are adverse",
            "side effects of",
            "risks of",
            "complications of",
            "adverse effects of",
            "adverse events of",
        ]
        if any(phrase in q for phrase in complication_trigger_phrases):
            return "complications"

        # ── Comparison ────────────────────────────────────────────────────────
        comparison_trigger_phrases = [" vs ", " versus ", "difference between", "compare "]
        if any(phrase in q for phrase in comparison_trigger_phrases):
            return "comparison"

        # ── Timing ────────────────────────────────────────────────────────────
        timing_trigger_phrases = [
            "when should", "when is", "when to ", "when do ",
            "when are", "when would",
        ]
        if any(q.startswith(phrase) or f" {phrase}" in q for phrase in timing_trigger_phrases):
            return "timing"

        # Not confident — let LLM decide
        return None

    def _reclassify_if_diagnostic_tool(self, question: str, original_category: str) -> str:
        """
        Detect if a 'definition' question is actually about a diagnostic tool/procedure.
        If so, reclassify as 'diagnosis'.
        """
        q_lower = question.lower()

        diagnostic_tools = [
            'ultrasound', 'ultrasonography', 'us', 'sonography',
            'fnab', 'fna', 'fine-needle', 'fine needle', 'biopsy',
            'imaging', 'scan', 'ct', 'mri', 'pet', 'pet scan',
            'elastography', 'doppler',
            'thyroglobulin', 'calcitonin', 'tsh', 'biomarker',
            'molecular testing', 'genetic testing',
            'laryngoscopy', 'endoscopy'
        ]

        role_indicators = [
            'role of', 'role in',
            'use of', 'use in', 'used for', 'used in',
            'purpose of', 'utility of', 'value of',
            'how is', 'how does', 'how can',
            'evaluate', 'evaluating', 'evaluation',
            'assess', 'assessing', 'assessment',
            'detect', 'detecting', 'detection'
        ]

        mentions_tool = any(tool in q_lower for tool in diagnostic_tools)
        mentions_role = any(indicator in q_lower for indicator in role_indicators)

        if mentions_tool and (mentions_role or 'what is' in q_lower or 'what does' in q_lower):
            logger.info(f"Reclassifying '{question}' from 'definition' to 'diagnosis' (diagnostic tool detected)")
            return "diagnosis"

        return original_category

    def _reclassify_if_evidence_question(self, question: str, original_category: str) -> str:
        """
        Post-LLM safety net for evidence questions.

        Mirrors the existing _reclassify_if_diagnostic_tool pattern.
        If the LLM returned 'definition' but the question is clearly about
        evidence, trials, or efficacy — override to 'evidence'.
        """
        q = question.lower()

        # Strong phrases → always override
        strong_evidence_phrases = [
            "what is the evidence",
            "evidence for",
            "evidence of",
            "evidence on",
            "is there evidence",
            "what trials",
            "efficacy of",
            "how effective is",
            "what studies",
            "what research",
            "clinical evidence",
            "trial data",
        ]
        if any(phrase in q for phrase in strong_evidence_phrases):
            logger.info(
                f"Reclassifying '{question}' from 'definition' → 'evidence' "
                f"(strong evidence phrase matched in post-LLM check)"
            )
            return "evidence"

        # Weak indicators — only override if 2+ present together
        weak_evidence_indicators = [
            "evidence", "trial", "study", "studies", "research",
            "efficacy", "effective", "outcome", "outcomes",
            "survival", "response rate", "pfs", "os",
            "randomized", "systematic review", "meta-analysis",
            "phase 3", "phase 2", "rct",
        ]
        matches = sum(1 for indicator in weak_evidence_indicators if indicator in q)
        if matches >= 2:
            logger.info(
                f"Reclassifying '{question}' from 'definition' → 'evidence' "
                f"({matches} weak evidence indicators matched)"
            )
            return "evidence"

        return original_category

    # =========================================================================
    # RETRIEVAL HELPERS
    # =========================================================================

    def _rerank_with_cross_encoder(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        top_k: int = SECOND_STAGE_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using cross-encoder for better relevance.
        """
        if not chunks:
            return []

        logger.info(f"Re-ranking {len(chunks)} chunks with cross-encoder...")

        pairs = []
        for chunk in chunks:
            doc_text = chunk.get("text", "")
            title = chunk.get("title", "")
            if title and title not in doc_text:
                combined = f"{title}. {doc_text}"
            else:
                combined = doc_text
            combined = combined[:2000]
            pairs.append([question, combined])

        scores = self.cross_encoder.predict(pairs)

        for idx, chunk in enumerate(chunks):
            chunk["score"] = float(scores[idx])

        reranked = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        top_chunks = reranked[:top_k]

        logger.info(
            f"Re-ranking complete: "
            f"Top score={top_chunks[0]['score']:.4f}, "
            f"Bottom score={top_chunks[-1]['score']:.4f}"
        )

        for i, chunk in enumerate(top_chunks[:3], 1):
            logger.info(
                f"  Rank {i}: score={chunk['score']:.4f}, "
                f"title='{chunk.get('title', 'N/A')[:60]}...'"
            )

        return top_chunks

    def diagnose_retrieval(self, question: str, k: int = 10) -> Dict[str, Any]:
        """
        Diagnostic tool to see what's actually being retrieved.
        """
        logger.info("=== DIAGNOSTIC MODE ===")

        sub_queries = self._expand_query_with_llm(question)

        diagnosis = {
            "original_question": question,
            "sub_queries_generated": sub_queries,
            "retrieval_results": []
        }

        for idx, sub_query in enumerate(sub_queries, 1):
            logger.info(f"Testing sub-query {idx}/{len(sub_queries)}: {sub_query}")

            chunks = self.vector_store.search(sub_query, k=k)

            result = {
                "query": sub_query,
                "chunks_found": len(chunks),
                "sample_chunks": []
            }

            for i, chunk in enumerate(chunks[:3], 1):
                result["sample_chunks"].append({
                    "rank": i,
                    "title": chunk.get("title", "No title"),
                    "year": chunk.get("year", "Unknown"),
                    "pmid": chunk.get("pmid", "Unknown"),
                    "evidence_level": chunk.get("evidence_level", "Unknown"),
                    "score": chunk.get("score", 0.0),
                    "text_preview": chunk.get("text", "")[:400] + "..."
                })

            diagnosis["retrieval_results"].append(result)

        logger.info("=== END DIAGNOSTIC ===")
        return diagnosis

    def _expand_query_with_llm(self, question: str) -> List[str]:
        """
        Use LLM to intelligently expand the query into multiple sub-queries.
        """
        expansion_prompt = f"""You are a medical information retrieval assistant specialized in thyroid cancer. Given a user's question, generate 3-5 targeted search queries that will retrieve comprehensive information from medical literature.

IMPORTANT GUIDELINES:
1. Include the original question
2. Generate queries focusing on:
   - Main topic (procedures, treatments, diagnoses)
   - Complications, adverse effects, and risks
   - Clinical outcomes and prognosis
   - Patient selection and indications
   - Alternative approaches or management strategies

3. Use specific medical terminology that would appear in research papers

EXAMPLES:

Question: "What are the standard surgical options for differentiated thyroid cancer?"
Queries: [
  "standard surgical options differentiated thyroid cancer",
  "complications risks thyroidectomy differentiated thyroid cancer",
  "lobectomy versus total thyroidectomy outcomes",
  "patient selection criteria thyroid surgery"
]

Question: "What are the complications of radioactive iodine therapy?"
Queries: [
  "complications radioactive iodine therapy thyroid cancer",
  "adverse effects RAI treatment",
  "toxicity side effects iodine-131 therapy",
  "late effects radioiodine ablation",
  "salivary gland dysfunction xerostomia RAI"
]

NOW GENERATE QUERIES FOR THIS QUESTION:
{question}

Return ONLY a JSON array of 3-5 search queries, no other text:"""

        try:
            logger.info("Expanding query with LLM...")
            response = self.llm.ask(expansion_prompt)

            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```\s*$', '', cleaned)

            queries = json.loads(cleaned)

            if isinstance(queries, list) and len(queries) > 0:
                if question not in queries:
                    queries.insert(0, question)

                queries = self._add_fallback_queries(question, queries)
                logger.info(f"Expanded into {len(queries)} queries: {queries}")
                return queries
            else:
                logger.warning("LLM returned invalid query expansion, using fallback")
                return self._create_fallback_queries(question)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM query expansion JSON: {e}")
            return self._create_fallback_queries(question)
        except Exception as e:
            logger.error(f"Error during query expansion: {e}")
            return self._create_fallback_queries(question)

    def _add_fallback_queries(self, original_question: str, existing_queries: List[str]) -> List[str]:
        """Add domain-specific fallback queries based on question type."""
        q_lower = original_question.lower()
        additional = []

        if any(word in q_lower for word in ['complication', 'risk', 'side effect', 'adverse', 'toxicity']):
            topic = self._extract_topic(original_question)
            if topic:
                additional.extend([
                    f"adverse effects {topic}",
                    f"toxicity {topic}",
                    f"late complications {topic}",
                ])

        elif any(word in q_lower for word in ['evidence', 'trial', 'efficacy', 'outcome', 'study', 'data', 'effective']):
            # Evidence-specific expansion — targets trial data in the vector store
            topic = self._extract_topic(original_question) or "thyroid cancer treatment"
            if topic:
                additional.extend([
                    f"randomized controlled trial {topic}",
                    f"phase 3 trial {topic} progression-free survival",
                    f"clinical outcomes {topic} thyroid cancer",
                    f"FDA approval {topic} thyroid cancer",
                    f"real-world data {topic} thyroid cancer",
                ])

        elif any(word in q_lower for word in ['treatment', 'therapy', 'surgical', 'surgery', 'procedure']):
            topic = self._extract_topic(original_question)
            if topic and not any('complication' in q.lower() or 'risk' in q.lower() for q in existing_queries):
                additional.extend([
                    f"complications {topic}",
                    f"adverse effects {topic}",
                ])

        if any(term in q_lower for term in ['radioactive iodine', 'rai', 'i-131', 'iodine-131', 'radioiodine']):
            additional.extend([
                "salivary gland dysfunction radioactive iodine",
                "xerostomia RAI therapy",
                "secondary malignancy radioiodine",
            ])

        for query in additional:
            if query not in existing_queries:
                existing_queries.append(query)

        return existing_queries

    def _extract_topic(self, question: str) -> Optional[str]:
        """Extract the main medical topic from the question."""
        q_lower = question.lower()

        patterns = [
            r'(?:of|for)\s+(.+?)(?:\?|$)',
            r'(?:what|how)\s+(?:is|are)\s+(.+?)(?:\?|treated|diagnosed)',
        ]

        for pattern in patterns:
            match = re.search(pattern, q_lower)
            if match:
                topic = match.group(1).strip()
                topic = re.sub(r'\s+(in|for|with)\s+thyroid.*', ' thyroid', topic)
                return topic

        return None

    def _create_fallback_queries(self, question: str) -> List[str]:
        """Create rule-based fallback queries when LLM expansion fails."""
        q_lower = question.lower()
        queries = [question]

        if any(word in q_lower for word in ['complication', 'risk', 'adverse', 'side effect']):
            topic = self._extract_topic(question) or "thyroid cancer treatment"
            queries.extend([
                f"complications {topic}",
                f"adverse effects {topic}",
                f"toxicity {topic}",
            ])
        elif any(word in q_lower for word in ['surgical', 'surgery', 'procedure', 'operation']):
            queries.extend([
                f"{question.replace('?', '')} complications",
                f"{question.replace('?', '')} risks",
            ])

        logger.info(f"Using fallback queries: {queries}")
        return queries

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks."""
        seen = set()
        unique = []

        for chunk in chunks:
            text_snippet = chunk.get("text", "")[:200].strip()
            pmid = chunk.get("pmid", "unknown")
            chunk_id = f"{pmid}||{hash(text_snippet)}"

            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk)

        logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique)} unique chunks")
        return unique

    def _build_tagged_context(self, retrieved: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict]]:
        """
        Build context with source tags for provenance tracking.
        Returns: (tagged_context, source_map)
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in retrieved:
            key = str(r.get("pmid") or r.get("title"))
            grouped.setdefault(key, []).append(r)

        source_map = {}
        tagged_parts = []
        total_chars = 0
        source_id = 1

        for group in grouped.values():
            if source_id > MAX_SOURCES:
                break

            meta = group[0]
            source_tag = f"SOURCE_{source_id}"

            source_map[source_tag] = {
                "title": meta.get("title", "Unknown"),
                "year": meta.get("year", "Unknown"),
                "pmid": meta.get("pmid", "Unknown"),
                "evidence_level": meta.get("evidence_level", "Unknown"),
                "cross_encoder_score": meta.get("score", 0.0),
            }

            for chunk in group[:MAX_CHUNKS_PER_SOURCE]:
                text = chunk.get("text", "")[:MAX_EXCERPT_CHARS]
                tagged_text = f"[{source_tag}] {text}"

                if total_chars + len(tagged_text) > MAX_TOTAL_CONTEXT_CHARS:
                    break

                tagged_parts.append(tagged_text)
                total_chars += len(tagged_text)

            source_id += 1

        context = "\n\n".join(tagged_parts)
        logger.info(f"Built tagged context with {len(source_map)} sources")
        return context, source_map

    def _compute_confidence(self, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence based on evidence levels."""
        levels = [
            r.get("evidence_level")
            for r in retrieved
            if r.get("evidence_level") in EVIDENCE_LEVEL_WEIGHTS
        ]

        if not levels:
            return {"label": "Low", "score": 0, "breakdown": "No evidence metadata"}

        weights = [EVIDENCE_LEVEL_WEIGHTS[l][1] for l in levels]
        score = int(round((sum(weights) / len(weights)) * 100))

        if score >= 85:
            label = "High"
        elif score >= 65:
            label = "Medium"
        else:
            label = "Low"

        breakdown = "; ".join(
            f"Level {l} ({EVIDENCE_LEVEL_WEIGHTS[l][0]}): {levels.count(l)}"
            for l in sorted(set(levels))
        )

        return {"label": label, "score": score, "breakdown": breakdown}

    # =========================================================================
    # STEP 2 + STEP 4 — PROMPT TEMPLATES
    # =========================================================================

    def _create_type_specific_prompt(
        self,
        question: str,
        context: str,
        question_type: str
    ) -> str:
        """
        Build a question-type-specific prompt that returns structured JSON.

        Step 2: Added 'evidence' template — structured by cancer subtype,
                requires trial names and numerical outcomes, Key Considerations block.
        Step 4: Every template now has a STRICT SCHEMA CONTRACT that instructs
                the LLM to write null instead of inventing content, forbids
                placeholder text, and requires [SOURCE_X] tags on every claim.
        """

        base_instructions = f"""
{self.instructions}

You are a medical information assistant specialised in thyroid cancer.
Answer using ONLY the tagged excerpts below. Do NOT use any outside knowledge.

SOURCE TAGGING RULES:
- Every factual claim must end with its source tag: "Example fact [SOURCE_1]."
- Combine tags when merging facts from multiple sources: "Fact [SOURCE_1][SOURCE_3]."
- Never write a factual sentence without at least one [SOURCE_X] tag.

STRICT SCHEMA CONTRACT — READ BEFORE WRITING:
1. NEVER invent content. If a field has no supporting data in the context, set it to null.
2. NEVER use placeholder text like "[Topic]", "[Drug Name]", "[X]", "[Insert here]", "Type Name".
3. ALL string values must be real sentences from the context — not template examples.
4. Return ONLY valid JSON. No markdown fences, no explanation, no extra text.

CONTEXT WITH SOURCE TAGS:
{context}

"""

        # ── EVIDENCE (Step 2 — new template) ─────────────────────────────────
        if question_type == "evidence":
            return base_instructions + f"""
QUESTION: {question}

You are synthesising clinical trial evidence. Your job is NOT to summarise sources —
it is to construct a unified expert answer organised by cancer subtype, exactly like
a clinical guideline would present it.

EVIDENCE CONTRACT (Step 4 — required fields):
- Every drug item MUST include the trial name if one appears in the context
  (e.g. SELECT trial, DECISION trial, ZETA trial, EXAM trial).
- Every drug item MUST include at least one numerical outcome if one exists in
  the context (e.g. median PFS in months, hazard ratio, ORR percentage).
- If no trial name or number is available in the context for a drug, set
  that drug's "description" to null — NEVER invent statistics.
- Omit sections entirely (set to null) if the context has no data for that cancer type.
- The "Key Considerations" section is REQUIRED. It must have at least 2 items.
- Items in sections must be arrays even if there is only one item.

Return this exact JSON structure:

{{
  "overview": "2-3 sentences: what TKIs are, their general role in advanced thyroid cancer, approval status, overall evidence quality. [SOURCE_X] tags required. REQUIRED — never null.",

  "sections": [

    {{
      "header": "Evidence in Differentiated Thyroid Cancer (DTC)",
      "items": [
        {{
          "name": "Exact drug name from context (e.g. Lenvatinib, Sorafenib). REQUIRED. null only if not in context.",
          "description": "Trial name (e.g. SELECT trial) + primary endpoint with exact numbers (e.g. median PFS 18.3 vs 3.6 months, HR 0.21, p<0.001) + ORR if available + approval status. [SOURCE_X]. null if no numerical data in context.",
          "highlight": "One sentence on what makes this agent clinically significant or preferred. [SOURCE_X]. null if not in context."
        }}
      ]
    }},

    {{
      "header": "Evidence in Medullary Thyroid Cancer (MTC)",
      "items": [
        {{
          "name": "Exact drug name from context. null if not in context.",
          "description": "Trial name + primary endpoint numbers + mutation targets (e.g. RET) + approval status. [SOURCE_X]. null if no data.",
          "highlight": "One sentence on clinical positioning or mutation relevance. [SOURCE_X]. null if not in context."
        }}
      ]
    }},

    {{
      "header": "Evidence in Anaplastic Thyroid Cancer (ATC)",
      "items": [
        {{
          "name": "Exact drug or combination name from context (e.g. Dabrafenib + Trametinib). null if not in context.",
          "description": "Evidence strength (limited/emerging), key outcomes with numbers if available, why TKI monotherapy has limited efficacy if mentioned. [SOURCE_X]. null if no data.",
          "highlight": "Key limitation or reason combination therapy is preferred if mentioned. [SOURCE_X]. null if not in context."
        }}
      ]
    }},

    {{
      "header": "Key Considerations",
      "items": [
        {{
          "consideration": "Toxicity Profile",
          "description": "Rate of grade 3 or higher adverse events, most common AEs (e.g. hypertension, hand-foot syndrome, fatigue) with percentages if available in context. [SOURCE_X]. null if not in context."
        }},
        {{
          "consideration": "Resistance",
          "description": "When and why resistance develops, salvage options if mentioned in context. [SOURCE_X]. null if not in context."
        }},
        {{
          "consideration": "Patient Selection",
          "description": "Criteria for initiating TKI therapy: radioiodine-refractory, symptomatic, rapidly progressive disease, as stated in context. [SOURCE_X]. null if not in context."
        }},
        {{
          "consideration": "Treatment Sequencing",
          "description": "First-line vs second-line positioning, role of watchful waiting, as stated in context. [SOURCE_X]. null if not in context."
        }}
      ]
    }}

  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── DEFINITION ────────────────────────────────────────────────────────
        elif question_type == "definition":
            return base_instructions + f"""
QUESTION: {question}

DEFINITION CONTRACT (Step 4):
- Use actual type names from context (e.g. Papillary Thyroid Cancer, Follicular).
  NEVER write "Type Name" or "[Topic]" as a placeholder.
- If types are not discussed in context, set the Types items array to null.
- All section content must come from context. null if not present.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence definition: what it is, who it affects, general outlook. [SOURCE_X]. REQUIRED — never null.",

  "sections": [
    {{
      "header": "Types",
      "items": [
        {{
          "name": "Actual type name from context (e.g. Papillary Thyroid Cancer). null if not in context.",
          "description": "What distinguishes this type, incidence if mentioned. [SOURCE_X]. null if not in context.",
          "details": "Prognosis, survival rates, or key statistics if available. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Causes and Risk Factors",
      "content": "Paragraph on known causes and risk factors from context. [SOURCE_X]. null if not in context."
    }},
    {{
      "header": "Common Symptoms",
      "items": [
        {{
          "symptom": "Actual symptom name from context. null if not in context.",
          "description": "Explanation of how it presents or why it occurs. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Diagnosis and Treatment",
      "content": "Paragraph on how it is diagnosed and treated based on context. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── COMPLICATIONS ─────────────────────────────────────────────────────
        elif question_type == "complications":
            return base_instructions + f"""
QUESTION: {question}

COMPLICATIONS CONTRACT (Step 4):
- Use real complication names from context. NEVER write "Name" or "Complication" as placeholders.
- Include frequency/percentage data exactly as stated in context.
- Set any section with no supporting context data to null entirely.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence summary of the complication landscape for this procedure or treatment. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "Common and Temporary Complications",
      "items": [
        {{
          "complication": "Real complication name from context. null if not present.",
          "description": "Explanation, mechanism, and frequency if mentioned. [SOURCE_X]. null if not in context.",
          "frequency": "Percentage or rate exactly as stated in context (e.g. 30-40% of cases). null if not stated."
        }}
      ]
    }},
    {{
      "header": "Less Common or Potentially Permanent Complications",
      "items": [
        {{
          "complication": "Real complication name from context. null if not present.",
          "description": "Explanation and any long-term impact. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Rare but Serious Complications",
      "items": [
        {{
          "complication": "Real complication name from context. null if not present.",
          "description": "Why it is serious and how it is managed if mentioned. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Management and Prevention",
      "content": "Paragraph on how complications are managed or prevented based on context. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── COMPARISON ────────────────────────────────────────────────────────
        elif question_type == "comparison":
            return base_instructions + f"""
QUESTION: {question}

COMPARISON CONTRACT (Step 4):
- Replace option_a_label and option_b_label with the ACTUAL names of the two things
  being compared — extract them from the question itself.
- Only include comparison_table rows where context has data for BOTH options.
- Set sections to null if no supporting context data.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence comparison summary explaining what is being compared and the clinical context. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "Key Differences",
      "option_a_label": "Exact name of first option from the question.",
      "option_b_label": "Exact name of second option from the question.",
      "comparison_table": [
        {{
          "aspect": "Specific aspect being compared (e.g. Mechanism, PFS, Toxicity). null if not in context.",
          "option_a": "What the first option does for this aspect with numbers if available. [SOURCE_X]. null if not in context.",
          "option_b": "What the second option does for this aspect with numbers if available. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Shared Characteristics",
      "content": "Paragraph on what both options have in common. [SOURCE_X]. null if not in context."
    }},
    {{
      "header": "Clinical Outcomes",
      "content": "Paragraph comparing survival, response rates, or other outcomes with numbers if available. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── TREATMENT ─────────────────────────────────────────────────────────
        elif question_type == "treatment":
            return base_instructions + f"""
QUESTION: {question}

TREATMENT CONTRACT (Step 4):
- Use real treatment and drug names from context. NEVER write "Treatment Name" as a placeholder.
- Include evidence grade (RCT, guideline, observational) only if discernible from context.
- Set sections to null if no supporting context data exists for them.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence treatment landscape summary with guideline-recommended first-line approaches. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "First-Line Treatments",
      "items": [
        {{
          "treatment": "Real treatment or drug name from context. null if not present.",
          "description": "Indication, mechanism, key trial result with numbers if available, approval status. [SOURCE_X]. null if not in context.",
          "evidence_grade": "e.g. Phase 3 RCT / Guideline recommendation / Observational. null if not discernible."
        }}
      ]
    }},
    {{
      "header": "Second-Line and Salvage Treatments",
      "items": [
        {{
          "treatment": "Real treatment name from context. null if not present.",
          "description": "When used, outcomes, any key trial with numbers. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Treatment by Cancer Subtype",
      "content": "Paragraph mapping approved agents to DTC, MTC, ATC, and PDTC subtypes where context supports it. [SOURCE_X]. null if not in context."
    }},
    {{
      "header": "Key Considerations",
      "items": [
        {{
          "consideration": "Real consideration title from context (e.g. Monitoring, Resistance). null if not present.",
          "description": "Concise explanation with any data. [SOURCE_X]. null if not in context."
        }}
      ]
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── DIAGNOSIS ─────────────────────────────────────────────────────────
        elif question_type == "diagnosis":
            return base_instructions + f"""
QUESTION: {question}

DIAGNOSIS CONTRACT (Step 4):
- Use real procedure names from context (e.g. Ultrasound, FNAB, CT).
  NEVER write "Procedure Name" as a placeholder.
- Include accuracy/sensitivity/specificity only if stated in context.
- Set sections to null if no supporting context data.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence summary of the diagnostic approach and why it is used. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "Key Diagnostic Procedures",
      "items": [
        {{
          "procedure": "Real procedure name from context. null if not present.",
          "description": "What it does, when it is used, clinical role. [SOURCE_X]. null if not in context.",
          "accuracy": "Sensitivity, specificity, or accuracy percentage as stated in context. null if not stated."
        }}
      ]
    }},
    {{
      "header": "Diagnostic Pathway",
      "content": "Paragraph describing the step-by-step process from initial presentation through to diagnosis. [SOURCE_X]. null if not in context."
    }},
    {{
      "header": "Limitations and Considerations",
      "content": "Paragraph on known limitations of the diagnostic approach or tools. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── TIMING ────────────────────────────────────────────────────────────
        elif question_type == "timing":
            return base_instructions + f"""
QUESTION: {question}

TIMING CONTRACT (Step 4):
- Use real clinical situations and patient profiles from context.
  NEVER write "Situation" or "Factor name" as placeholders.
- Set sections to null if no supporting context data.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence summary of when and why this is recommended and what guides the decision. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "Key Indications",
      "items": [
        {{
          "indication": "Real clinical situation or patient profile from context. null if not present.",
          "explanation": "Why this timing is recommended in this case with any supporting data. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Important Considerations",
      "items": [
        {{
          "consideration": "Real factor affecting timing decisions from context (e.g. tumour size, patient age). null if not present.",
          "description": "How this factor influences the timing recommendation. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Contraindications or When to Delay",
      "content": "Paragraph on situations where this should be delayed or avoided based on context. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── FALLBACK ──────────────────────────────────────────────────────────
        else:
            logger.warning(f"Unknown question_type '{question_type}', falling back to definition")
            return self._create_type_specific_prompt(question, context, "definition")

    # =========================================================================
    # MAIN ANSWER METHOD
    # =========================================================================

    def answer(
        self,
        question: str,
        chat_history: Optional[list] = None,
        k: int = 30
    ) -> Dict[str, Any]:
        """
        Generate answer as structured JSON with source tracking.
        Pipeline: classify → expand → retrieve → deduplicate → rerank → answer → evaluate
        """
        # Step 1: Classify question type
        question_type = self._classify_question_type(question)

        # Step 2: Expand query
        sub_queries = self._expand_query_with_llm(question)

        # Step 3: First-stage retrieval (bi-encoder)
        logger.info("=== FIRST STAGE: Bi-encoder retrieval ===")
        all_retrieved = []
        chunks_per_query = FIRST_STAGE_RETRIEVAL // len(sub_queries)

        for idx, sub_query in enumerate(sub_queries, 1):
            logger.info(f"Retrieving for sub-query {idx}/{len(sub_queries)}: {sub_query}")
            retrieved = self.vector_store.search(sub_query, k=chunks_per_query)
            all_retrieved.extend(retrieved)

        logger.info(f"First stage retrieved {len(all_retrieved)} chunks")

        # Step 4: Deduplicate
        unique_retrieved = self._deduplicate_chunks(all_retrieved)

        if not unique_retrieved:
            return {
                "error": "No relevant information found",
                "json_response": None,
                "sources": {},
                "confidence": {"label": "Low", "score": 0, "breakdown": "No data"}
            }

        # Step 5: Second-stage re-ranking (cross-encoder)
        logger.info("=== SECOND STAGE: Cross-encoder re-ranking ===")
        reranked_chunks = self._rerank_with_cross_encoder(
            question=question,
            chunks=unique_retrieved,
            top_k=SECOND_STAGE_TOP_K
        )

        logger.info(f"Using top {len(reranked_chunks)} re-ranked chunks for context")

        # Step 6: Build tagged context
        context, source_map = self._build_tagged_context(reranked_chunks)

        # Step 7: Compute confidence
        confidence = self._compute_confidence(reranked_chunks)

        # Step 8: Generate JSON answer
        logger.info(f"Generating {question_type} answer with LLM...")
        prompt = self._create_type_specific_prompt(question, context, question_type)

        try:
            response = self.llm.ask(prompt).strip()

            # Clean JSON fences if present
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*\n?', '', response)
                response = re.sub(r'\n?```\s*$', '', response)

            json_response = json.loads(response)

            # Step 9: Evaluate faithfulness
            logger.info("Evaluating answer faithfulness...")
            try:
                faithfulness = self.faithfulness_evaluator.evaluate(
                    json_response=json_response,
                    tagged_context=context,
                    source_map=source_map
                )
                logger.info(
                    f"Faithfulness evaluation: {faithfulness.get('label', 'N/A')} "
                    f"({faithfulness.get('score', 'N/A')})"
                )
            except Exception as e:
                logger.error(f"Faithfulness evaluation failed: {e}", exc_info=True)
                faithfulness = {
                    "score": None,
                    "label": "Not Available",
                    "error": str(e),
                    "total_statements": 0,
                    "evaluated_statements": 0
                }

            return {
                "json_response": json_response,
                "sources": source_map,
                "confidence": confidence,
                "faithfulness": faithfulness,
                "question_type": question_type,
                "retrieval_stats": {
                    "first_stage_retrieved": len(all_retrieved),
                    "after_dedup": len(unique_retrieved),
                    "after_reranking": len(reranked_chunks),
                }
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"LLM response was: {response}")
            return {
                "error": f"Failed to generate structured response: {str(e)}",
                "json_response": None,
                "sources": source_map,
                "confidence": confidence
            }
