# rag/qa_pipeline.py
import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import CrossEncoder

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

    def _classify_question_type(self, question: str) -> str:
        """
        Classify the question type to use appropriate template.
        Returns: 'definition', 'complications', 'comparison', 'treatment', 'diagnosis', 'timing'
        """
        classification_prompt = f"""Classify this thyroid cancer question into ONE category:

Categories:
- definition: "What is X?", "Tell me about X", "Explain X"
- complications: "What are complications/risks/side effects of X?"
- comparison: "X vs Y", "difference between X and Y", "compare X and Y"
- treatment: "How to treat X?", "What are treatment options?", "How is X treated?"
- diagnosis: "How is X diagnosed?", "What tests for X?", "How to detect X?"
- timing: "When should X?", "When is X recommended?", "When to do X?"

Question: {question}

Return ONLY the category name (one word), nothing else:"""

        try:
            category = self.llm.ask(classification_prompt).strip().lower()
            # Validate category
            valid_categories = ["definition", "complications", "comparison", "treatment", "diagnosis", "timing"]
            if category in valid_categories:
                logger.info(f"Question classified as: {category}")
                return category
            else:
                logger.warning(f"Invalid category '{category}', defaulting to 'definition'")
                return "definition"
        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            return "definition"  # Default fallback

    def _rerank_with_cross_encoder(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = SECOND_STAGE_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using cross-encoder for better relevance.
        
        Args:
            question: The user's question
            chunks: List of candidate chunks from bi-encoder
            top_k: Number of top chunks to return after re-ranking
            
        Returns:
            Re-ranked list of chunks with updated scores
        """
        if not chunks:
            return []
        
        logger.info(f"Re-ranking {len(chunks)} chunks with cross-encoder...")
        
        # Prepare (question, document) pairs
        pairs = []
        for chunk in chunks:
            # Combine title and text for better context
            doc_text = chunk.get("text", "")
            title = chunk.get("title", "")
            
            # Create document representation
            if title and title not in doc_text:
                combined = f"{title}. {doc_text}"
            else:
                combined = doc_text
            
            # Truncate if too long (cross-encoder usually has ~512 token limit)
            combined = combined[:2000]  # roughly ~500 tokens
            pairs.append([question, combined])
        
        # Score all pairs
        scores = self.cross_encoder.predict(pairs)
        
        # Replace bi-encoder scores with cross-encoder scores
        for idx, chunk in enumerate(chunks):
            chunk["score"] = float(scores[idx])
        
        # Sort by cross-encoder score (descending)
        reranked = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        
        # Take top_k
        top_chunks = reranked[:top_k]
        
        logger.info(
            f"Re-ranking complete: "
            f"Top score={top_chunks[0]['score']:.4f}, "
            f"Bottom score={top_chunks[-1]['score']:.4f}"
        )
        
        # Log top 3 for debugging
        for i, chunk in enumerate(top_chunks[:3], 1):
            logger.info(
                f"  Rank {i}: score={chunk['score']:.4f}, "
                f"title='{chunk.get('title', 'N/A')[:60]}...'"
            )
        
        return top_chunks

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
            
            # Clean up response
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
        """Add domain-specific fallback queries."""
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
        # Group by source
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
            
            # Store source metadata
            source_map[source_tag] = {
                "title": meta.get("title", "Unknown"),
                "year": meta.get("year", "Unknown"),
                "pmid": meta.get("pmid", "Unknown"),
                "evidence_level": meta.get("evidence_level", "Unknown"),
                "cross_encoder_score": meta.get("score", 0.0),  # Add cross-encoder score
            }
            
            # Add tagged excerpts
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

    def _create_type_specific_prompt(
        self, 
        question: str, 
        context: str, 
        question_type: str
    ) -> str:
        """
        Create question-type-specific prompts that return structured JSON.
        """
        
        base_instructions = f"""
{self.instructions}

You are a medical information assistant specialized in thyroid cancer. Answer using ONLY the tagged excerpts below.

CRITICAL SOURCE TAGGING RULES:
1. Each excerpt is tagged with [SOURCE_X]
2. When you write a fact, include the source tag: "Papillary cancer is most common [SOURCE_1]."
3. Use multiple tags if combining sources: "Surgery is the main treatment [SOURCE_1][SOURCE_3]."
4. EVERY factual claim must have at least one source tag

CONTEXT WITH SOURCE TAGS:
{context}

"""

        # Type-specific JSON templates
        if question_type == "definition":
            return base_instructions + f"""
QUESTION: {question}

Return a JSON object with ONLY the sections that are relevant to answer this specific question.

GUIDELINES FOR SECTION INCLUSION:

1. "Types of [Topic]" - Include ONLY if:
   ✅ Question asks about types/subtypes of a DISEASE or CONDITION
   ✅ Examples: "What is thyroid cancer?" "What is DTC?"
   ❌ Skip if question is about: diagnostic tools, procedures, biomarkers, or "role/purpose"
   ❌ Examples to SKIP: "What is the role of ultrasound?" "What is FNAB used for?"

2. "Key Features" or "Clinical Applications" - Use for:
   ✅ Diagnostic tools (ultrasound, FNAB, imaging)
   ✅ Procedures or interventions
   ✅ "What is the role of X?" questions

3. "Causes and Risk Factors" - Include if:
   ✅ Question is about a disease/condition with known causes
   ❌ Skip if question is about diagnostic tools or procedures

4. "Common Symptoms" - Include if:
   ✅ Question is about a disease/condition with symptoms
   ❌ Skip if question is about diagnostic procedures or lab tests

5. "Diagnosis and Treatment" - Include if relevant to the question

STRUCTURE:
{{
  "overview": "2-3 sentence answer with [SOURCE_X] tags",
  "sections": [
    // Only include sections that help answer THIS specific question
    // Use appropriate headers based on the question context
    {{
      "header": "Appropriate Section Header",
      "items": [...] OR "content": "..."
    }}
  ]
}}

EXAMPLES:

Q: "What is papillary thyroid cancer?"
→ Include: Types, Causes, Symptoms, Diagnosis ✅

Q: "What is the role of ultrasound in thyroid nodules?"
→ Include: Key Features, Clinical Applications, Diagnostic Pathway ✅
→ Skip: Types (ultrasound isn't a disease with subtypes) ❌

Return ONLY valid JSON, no other text:"""

        elif question_type == "complications":
            return base_instructions + f"""
QUESTION: {question}

Return a JSON object with relevant complication categories for this question.

GUIDELINES:

1. Group complications by severity/frequency if information is available:
   - Common & Temporary
   - Less Common & Potentially Permanent
   - Rare but Serious

2. If grouping isn't clear from sources, use a single "Complications" section

3. Include "Management & Prevention" if sources discuss it

4. Skip sections that aren't supported by the source material

STRUCTURE:
{{
  "overview": "2-3 sentence summary with [SOURCE_X] tags",
  "sections": [
    {{
      "header": "Appropriate Category",
      "items": [
        {{
          "complication": "Name",
          "description": "Full explanation with [SOURCE_X] tags",
          "frequency": "How common (if mentioned, otherwise omit)"
        }}
      ]
    }}
    // Add more sections as needed
  ]
}}

Return ONLY valid JSON, no other text:"""

        elif question_type == "comparison":
            return base_instructions + f"""
QUESTION: {question}

Return a JSON object comparing the items in the question.

GUIDELINES:

1. Include "Key Differences" section with comparison table OR bullet points
2. Include "Shared Characteristics" if relevant
3. Include "Outcomes" or "Clinical Implications" if discussed in sources
4. Skip sections not supported by source material

STRUCTURE:
{{
  "overview": "2-3 sentence comparison summary with [SOURCE_X] tags",
  "sections": [
    {{
      "header": "Key Differences",
      "comparison_table": [
        {{
          "aspect": "Aspect being compared",
          "option_a": "Description with [SOURCE_X]",
          "option_b": "Description with [SOURCE_X]"
        }}
      ]
      // OR use "items" for bullet point format if table doesn't fit
    }},
    {{
      "header": "Shared Characteristics",
      "content": "Paragraph with [SOURCE_X] tags"
    }}
    // Add other relevant sections
  ]
}}

Return ONLY valid JSON, no other text:"""

        elif question_type == "treatment":
            return base_instructions + f"""
QUESTION: {question}

Return a JSON object with relevant treatment information for this question.

GUIDELINES:

1. Group treatments logically (e.g., by type, by cancer stage, by line of therapy)
2. Common categories: "Standard Treatments", "Advanced/Emerging Treatments", "Treatment by Stage"
3. Include "Treatment Selection" or "Guidelines" if discussed
4. Skip categories not supported by sources

STRUCTURE:
{{
  "overview": "2-3 sentence summary with [SOURCE_X] tags",
  "sections": [
    {{
      "header": "Appropriate Category",
      "items": [
        {{
          "treatment": "Treatment name",
          "description": "Full explanation with [SOURCE_X] tags"
        }}
      ]
      // OR use "content" for paragraph format
    }}
    // Add relevant sections
  ]
}}

Return ONLY valid JSON, no other text:"""

        elif question_type == "diagnosis":
            return base_instructions + f"""
QUESTION: {question}

Return a JSON object with relevant diagnostic information for this question.

GUIDELINES:

1. For questions about diagnostic tools: focus on their clinical applications, key features, what they detect
2. For questions about diagnostic process: include pathway, procedures, and interpretation
3. Include accuracy/performance metrics if mentioned in sources
4. Skip sections not supported by sources

STRUCTURE:
{{
  "overview": "2-3 sentence summary with [SOURCE_X] tags",
  "sections": [
    {{
      "header": "Key Diagnostic Procedures" OR "Clinical Applications" OR appropriate header,
      "items": [
        {{
          "procedure": "Procedure name" OR "feature": "Feature name",
          "description": "What it does, when used, with [SOURCE_X] tags",
          "accuracy": "Accuracy info if mentioned"
        }}
      ]
      // OR use "content" for paragraph format
    }}
    // Add relevant sections like "Diagnostic Pathway", "Interpretation", etc.
  ]
}}

Return ONLY valid JSON, no other text:"""

        elif question_type == "timing":
            return base_instructions + f"""
QUESTION: {question}

Return a JSON object with relevant timing/indication information for this question.

GUIDELINES:

1. Include "Key Indications" when sources discuss when something should be done
2. Include "Important Considerations" or "Contraindications" if discussed
3. Include "Guidelines" if specific recommendations exist
4. Skip sections not supported by sources

STRUCTURE:
{{
  "overview": "2-3 sentence summary with [SOURCE_X] tags",
  "sections": [
    {{
      "header": "Key Indications" OR "When Recommended",
      "items": [
        {{
          "indication": "Situation/condition",
          "explanation": "Why recommended with [SOURCE_X] tags"
        }}
      ]
      // OR use "content" for paragraph format
    }}
    // Add relevant sections
  ]
}}

Return ONLY valid JSON, no other text:"""

        else:  # Fallback to definition
            return self._create_type_specific_prompt(question, context, "definition")

    def answer(
        self, 
        question: str, 
        chat_history: Optional[list] = None, 
        k: int = 30
    ) -> Dict[str, Any]:
        """
        Generate answer as structured JSON with source tracking.
        NOW WITH CROSS-ENCODER RE-RANKING!
        
        Returns:
            Dict with 'json_response', 'sources', 'confidence' keys
        """
        # Step 1: Classify question type
        question_type = self._classify_question_type(question)
        
        # Step 2: Expand query
        sub_queries = self._expand_query_with_llm(question)
        
        # Step 3: First-stage retrieval (bi-encoder) - retrieve MORE candidates
        logger.info(f"=== FIRST STAGE: Bi-encoder retrieval ===")
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

        # Step 5: Second-stage re-ranking (cross-encoder) - get BEST chunks
        logger.info(f"=== SECOND STAGE: Cross-encoder re-ranking ===")
        reranked_chunks = self._rerank_with_cross_encoder(
            question=question,
            chunks=unique_retrieved,
            top_k=SECOND_STAGE_TOP_K
        )
        
        logger.info(f"Using top {len(reranked_chunks)} re-ranked chunks for context")
        
        # Step 6: Build tagged context with re-ranked chunks
        context, source_map = self._build_tagged_context(reranked_chunks)
        
        # Step 7: Compute confidence
        confidence = self._compute_confidence(reranked_chunks)

        # Step 8: Generate JSON answer
        logger.info(f"Generating {question_type} answer with LLM...")
        prompt = self._create_type_specific_prompt(question, context, question_type)
        
        try:
            response = self.llm.ask(prompt).strip()
            
            # Clean JSON
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*\n?', '', response)
                response = re.sub(r'\n?```\s*$', '', response)
            
            json_response = json.loads(response)
            
            return {
                "json_response": json_response,
                "sources": source_map,
                "confidence": confidence,
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
