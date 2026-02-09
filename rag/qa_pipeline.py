# rag/qa_pipeline.py
import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Context building limits - INCREASED for better coverage
MAX_SOURCES = 8
MAX_CHUNKS_PER_SOURCE = 3
MAX_EXCERPT_CHARS = 1200
MAX_TOTAL_CONTEXT_CHARS = 8500


class QAPipeline:
    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        llm_client: Any,
        instruction_file: str = "instructions/rag_instructions.txt",
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm_client

        # Load instructions
        env = os.getenv("ENV", "local").lower()
        base = Path(__file__).parent if env == "prod" else Path(".")
        self.instructions = (base / instruction_file).read_text(encoding="utf-8")

    def diagnose_retrieval(self, question: str, k: int = 10) -> Dict[str, Any]:
        """
        Diagnostic tool to see what's actually being retrieved.
        Returns detailed info about retrieved chunks.
        """
        logger.info("=== DIAGNOSTIC MODE ===")
        
        # Generate sub-queries
        sub_queries = self._expand_query_with_llm(question)
        
        diagnosis = {
            "original_question": question,
            "sub_queries_generated": sub_queries,
            "retrieval_results": []
        }
        
        # Test each sub-query
        for idx, sub_query in enumerate(sub_queries, 1):
            logger.info(f"Testing sub-query {idx}/{len(sub_queries)}: {sub_query}")
            
            chunks = self.vector_store.search(sub_query, k=k)
            
            result = {
                "query": sub_query,
                "chunks_found": len(chunks),
                "sample_chunks": []
            }
            
            # Show first 3 chunks with full details
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
        Use LLM to intelligently expand the query into multiple sub-queries
        to capture different aspects (procedures, risks, outcomes, etc.)
        """
        expansion_prompt = f"""You are a medical information retrieval assistant specialized in thyroid cancer. Given a user's question, generate 3-5 targeted search queries that will retrieve comprehensive information from medical literature.

IMPORTANT GUIDELINES:
1. Include the original question
2. Generate queries focusing on:
   - Main topic (procedures, treatments, diagnoses)
   - Complications, adverse effects, and risks (use terms: "complications", "adverse effects", "toxicity", "side effects", "morbidity", "mortality")
   - Clinical outcomes and prognosis
   - Patient selection and indications
   - Alternative approaches or management strategies

3. Use specific medical terminology that would appear in research papers
4. For complications/risks questions, use multiple clinical terms (e.g., "adverse events", "toxicity", "late effects", "sequelae")

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

Question: "How is papillary thyroid cancer diagnosed?"
Queries: [
  "diagnosis papillary thyroid cancer",
  "fine needle aspiration biopsy thyroid",
  "imaging ultrasound papillary thyroid carcinoma",
  "molecular markers BRAF papillary thyroid cancer"
]

NOW GENERATE QUERIES FOR THIS QUESTION:
{question}

Return ONLY a JSON array of 3-5 search queries, no other text:"""

        try:
            logger.info("Expanding query with LLM...")
            response = self.llm.ask(expansion_prompt)
            
            # Clean up response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove ```json and ``` markers
                cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            
            # Parse JSON
            queries = json.loads(cleaned)
            
            if isinstance(queries, list) and len(queries) > 0:
                # Ensure original question is included
                if question not in queries:
                    queries.insert(0, question)
                
                # Add fallback specific queries for common topics
                queries = self._add_fallback_queries(question, queries)
                
                logger.info(f"Expanded into {len(queries)} queries: {queries}")
                return queries
            else:
                logger.warning("LLM returned invalid query expansion, using fallback")
                return self._create_fallback_queries(question)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM query expansion JSON: {e}")
            logger.error(f"LLM response was: {response}")
            return self._create_fallback_queries(question)
        except Exception as e:
            logger.error(f"Error during query expansion: {e}")
            return self._create_fallback_queries(question)

    def _add_fallback_queries(self, original_question: str, existing_queries: List[str]) -> List[str]:
        """
        Add domain-specific fallback queries based on keywords in the question.
        Ensures we search for complications/risks even if LLM doesn't generate them.
        """
        q_lower = original_question.lower()
        additional = []
        
        # If asking about complications/risks/side effects
        if any(word in q_lower for word in ['complication', 'risk', 'side effect', 'adverse', 'toxicity']):
            # Add multiple variations
            topic = self._extract_topic(original_question)
            if topic:
                additional.extend([
                    f"adverse effects {topic}",
                    f"toxicity {topic}",
                    f"late complications {topic}",
                    f"morbidity mortality {topic}",
                ])
        
        # If asking about treatment/surgery but NOT explicitly about complications
        elif any(word in q_lower for word in ['treatment', 'therapy', 'surgical', 'surgery', 'procedure']):
            topic = self._extract_topic(original_question)
            if topic and not any('complication' in q.lower() or 'risk' in q.lower() for q in existing_queries):
                additional.extend([
                    f"complications {topic}",
                    f"adverse effects {topic}",
                ])
        
        # If asking about radioactive iodine specifically
        if any(term in q_lower for term in ['radioactive iodine', 'rai', 'i-131', 'iodine-131', 'radioiodine']):
            additional.extend([
                "salivary gland dysfunction radioactive iodine",
                "xerostomia RAI therapy",
                "secondary malignancy radioiodine",
                "bone marrow suppression iodine-131",
            ])
        
        # Deduplicate and add
        for query in additional:
            if query not in existing_queries:
                existing_queries.append(query)
        
        return existing_queries

    def _extract_topic(self, question: str) -> Optional[str]:
        """Extract the main medical topic from the question."""
        q_lower = question.lower()
        
        # Common patterns
        patterns = [
            r'(?:of|for)\s+(.+?)(?:\?|$)',  # "complications of X" or "treatment for X"
            r'(?:what|how)\s+(?:is|are)\s+(.+?)(?:\?|treated|diagnosed)',  # "What is X" or "How is X treated"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, q_lower)
            if match:
                topic = match.group(1).strip()
                # Clean up
                topic = re.sub(r'\s+(in|for|with)\s+thyroid.*', ' thyroid', topic)
                return topic
        
        return None

    def _create_fallback_queries(self, question: str) -> List[str]:
        """
        Create rule-based fallback queries when LLM expansion fails.
        """
        q_lower = question.lower()
        queries = [question]  # Always include original
        
        # Detect question type and add relevant queries
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
                "thyroidectomy complications adverse effects",
            ])
        elif any(word in q_lower for word in ['treatment', 'therapy', 'manage']):
            queries.extend([
                f"{question.replace('?', '')} outcomes",
                f"{question.replace('?', '')} complications",
            ])
        
        logger.info(f"Using fallback queries: {queries}")
        return queries

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks based on text content and source.
        Uses first 200 characters of text for similarity detection.
        """
        seen = set()
        unique = []
        
        for chunk in chunks:
            # Create unique identifier from PMID + text snippet
            text_snippet = chunk.get("text", "")[:200].strip()
            pmid = chunk.get("pmid", "unknown")
            chunk_id = f"{pmid}||{hash(text_snippet)}"
            
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk)
        
        logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique)} unique chunks")
        return unique

    def _build_context(self, retrieved: List[Dict[str, Any]]) -> str:
        """Build context from retrieved chunks, grouped by source."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in retrieved:
            key = str(r.get("pmid") or r.get("title"))
            grouped.setdefault(key, []).append(r)

        parts = []
        total_chars = 0

        for idx, group in enumerate(grouped.values(), start=1):
            if idx > MAX_SOURCES:
                break
            
            meta = group[0]
            header = (
                f"SOURCE {idx}: {meta.get('title')} ({meta.get('year')})\n"
                f"PMID: {meta.get('pmid')} | Evidence Level: {meta.get('evidence_level')}\n"
            )
            
            if total_chars + len(header) > MAX_TOTAL_CONTEXT_CHARS:
                break
                
            parts.append(header)
            total_chars += len(header)

            # Add excerpts from this source
            for chunk in group[:MAX_CHUNKS_PER_SOURCE]:
                text = chunk.get("text", "")[:MAX_EXCERPT_CHARS]
                excerpt = f"EXCERPT:\n{text}\n\n"
                
                if total_chars + len(excerpt) > MAX_TOTAL_CONTEXT_CHARS:
                    break
                    
                parts.append(excerpt)
                total_chars += len(excerpt)

        return "".join(parts)

    def _log_context_preview(self, context: str) -> None:
        """Log first and last parts of context for debugging."""
        lines = context.split('\n')
        preview_lines = 20
        
        logger.info("=== CONTEXT PREVIEW (First 20 lines) ===")
        for line in lines[:preview_lines]:
            logger.info(line)
        logger.info("...")
        logger.info(f"=== CONTEXT PREVIEW (Last 20 lines of {len(lines)} total) ===")
        for line in lines[-preview_lines:]:
            logger.info(line)
        logger.info("=== END CONTEXT PREVIEW ===")

    def _compute_confidence(self, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence based on evidence levels."""
        levels = [
            r.get("evidence_level") 
            for r in retrieved 
            if r.get("evidence_level") in EVIDENCE_LEVEL_WEIGHTS
        ]
        
        if not levels:
            return {"label": "Low", "score": 0, "breakdown": "No evidence metadata"}

        # Calculate weighted score
        weights = [EVIDENCE_LEVEL_WEIGHTS[l][1] for l in levels]
        score = int(round((sum(weights) / len(weights)) * 100))
        
        # Determine label
        if score >= 85:
            label = "High"
        elif score >= 65:
            label = "Medium"
        else:
            label = "Low"

        # Create breakdown
        breakdown = "; ".join(
            f"Level {l} ({EVIDENCE_LEVEL_WEIGHTS[l][0]}): {levels.count(l)}"
            for l in sorted(set(levels))
        )

        return {"label": label, "score": score, "breakdown": breakdown}

    def _create_prompt(self, question: str, context: str) -> str:
        """Create Google-style overview prompt with VERY aggressive extraction instructions."""
        return f"""
{self.instructions}

You are a medical information assistant specialized in thyroid cancer. Your task is to provide clear, patient-friendly answers in the style of Google's AI Overview.

=== CRITICAL EXTRACTION RULES - READ CAREFULLY ===

1. Use ONLY information from the provided excerpts below
2. Do NOT cite sources inline (no parenthetical citations)
3. **EXTRACT EVERY RELEVANT DETAIL**: If an excerpt mentions a complication, side effect, risk, symptom, or adverse event even ONCE, you MUST include it in your answer
4. **DO NOT require "comprehensive" or "complete" lists** - report whatever you find
5. **NEVER say**: "not explicitly provided", "not exhaustively enumerated", "not uniformly listed", or "limited information"
6. **SEARCH EVERYWHERE**: Complications may be buried in:
   - Study results sections
   - Reference lists (e.g., "Smith et al reported xerostomia")
   - Case report descriptions
   - Discussion sections
   - Safety guideline recommendations
   - Passing mentions anywhere in the text

=== SPECIFIC INSTRUCTIONS FOR COMPLICATIONS/ADVERSE EFFECTS ===

When asked about complications, risks, or side effects:
✅ DO THIS:
- List EVERY complication mentioned in ANY excerpt (even if mentioned just once)
- Include percentages/frequencies (e.g., "5-86% of patients")
- Include descriptors (e.g., "dose-dependent", "rare", "common")
- Include mentions from references (e.g., "studies report taste impairment")
- Include case report findings (individual cases are valid data)
- Combine information from multiple excerpts

❌ DO NOT DO THIS:
- Say "not provided" when ANY information exists
- Require complete lists before reporting findings
- Ignore mentions in references or case reports
- Use hedging language like "may not be fully described"

=== EXAMPLES OF PROPER EXTRACTION ===

If excerpts contain:
✅ "may cause xerostomia" → **Include**: "Xerostomia (dry mouth)"
✅ "complications include taste impairment, sialadenitis" → **Include both**
✅ "salivary damage occurs in 5-86%" → **Include**: "Salivary dysfunction (5-86%)"
✅ "Smith reported leukemia risk" → **Include**: "Secondary leukemia (case reports)"
✅ "prevention includes lemon candy" → **Include in Prevention section**

=== OUTPUT FORMAT ===

**AI Overview**
[Write 1-2 specific paragraphs summarizing ALL complications found. Include numbers/percentages.]

**Known Complications/Adverse Effects:**
- **[Complication Name]**: [Full details including frequency, severity, timing if mentioned]
- **[Complication Name]**: [Full details]
- **[Complication Name]**: [Full details]
[LIST EVERY SINGLE COMPLICATION FOUND - do not limit yourself]

**Incidence/Risk Factors:**
- [Any frequency data, dose-dependence, risk factors mentioned]
[Only if information exists in excerpts]

**Prevention/Management Strategies:**
- [Any preventive measures, treatments, or management mentioned]
[Only if information exists in excerpts]

**Additional Considerations:**
- [Other relevant clinical information from excerpts]
[Only if information exists in excerpts]

=== YOUR TASK ===

QUESTION: {question}

CONTEXT FROM MEDICAL LITERATURE:
{context}

INSTRUCTIONS FOR THIS SPECIFIC ANSWER:
1. Read through EVERY excerpt above carefully - do not skip any
2. Extract EVERY mention of complications, adverse effects, risks, or side effects
3. Include specific details (percentages, frequencies, case counts)
4. Format as shown above
5. Be comprehensive - if you found 10 complications, list all 10

Begin your answer now:
""".strip()

    def _extract_sources(self, retrieved: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources for citation."""
        sources = []
        seen = set()
        
        for r in retrieved:
            title = r.get("title", "Unknown")
            year = r.get("year", "")
            key = (title, year)
            
            if key not in seen and title != "Unknown":
                seen.add(key)
                sources.append(f"• {title} ({year})")
                
            if len(sources) >= MAX_SOURCES:
                break
                
        return sources

    def answer(
        self, 
        question: str, 
        chat_history: Optional[list] = None, 
        k: int = 30
    ) -> str:
        """
        Generate a Google-style overview answer to the question.
        Uses LLM-powered query expansion for comprehensive retrieval.
        
        Args:
            question: User's question
            chat_history: Optional conversation history (not currently used)
            k: Total number of chunks to retrieve (distributed across sub-queries)
            
        Returns:
            Formatted answer with sources and evidence quality
        """
        # Step 1: Expand query into multiple sub-queries using LLM
        sub_queries = self._expand_query_with_llm(question)
        
        # Step 2: Retrieve chunks for each sub-query
        all_retrieved = []
        chunks_per_query = max(k // len(sub_queries), 8)  # At least 8 per query
        
        for idx, sub_query in enumerate(sub_queries, 1):
            logger.info(f"Retrieving for sub-query {idx}/{len(sub_queries)}: {sub_query}")
            retrieved = self.vector_store.search(sub_query, k=chunks_per_query)
            all_retrieved.extend(retrieved)
            logger.info(f"  Retrieved {len(retrieved)} chunks")
        
        # Step 3: Deduplicate chunks
        unique_retrieved = self._deduplicate_chunks(all_retrieved)
        
        if not unique_retrieved:
            return "I don't have enough information in my knowledge base to answer this question about thyroid cancer."

        # Step 4: Build context and compute confidence
        context = self._build_context(unique_retrieved)
        
        # Log context preview for debugging (comment out in production)
        self._log_context_preview(context)
        
        confidence = self._compute_confidence(unique_retrieved)

        # Step 5: Generate answer
        logger.info("Generating final answer with LLM...")
        prompt = self._create_prompt(question, context)
        answer = self.llm.ask(prompt).strip()

        # Step 6: Add evidence summary and sources
        sources = self._extract_sources(unique_retrieved)
        
        result = f"""{answer}

---

**Evidence Quality:** {confidence['label']} confidence ({confidence['score']}/100)
*Based on: {confidence['breakdown']}*

**Key Sources:**
{chr(10).join(sources) if sources else "• No sources available"}
""".strip()

        return result
