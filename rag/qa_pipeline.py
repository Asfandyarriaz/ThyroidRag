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

# Context building limits
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

    def _get_few_shot_examples(self) -> str:
        """
        Return curated examples of high-quality answers for different question types.
        These teach the LLM how to adapt its format to the question.
        """
        return """
=== EXAMPLE 1: DEFINITION QUESTION ===

Question: "What is thyroid cancer?"

Answer:
**AI Overview**
Thyroid cancer is a malignancy that develops in the thyroid gland, a butterfly-shaped organ at the base of the neck that produces hormones regulating metabolism. It's the most common endocrine cancer, typically affecting women between ages 30-60. Most thyroid cancers are slow-growing and highly treatable, with excellent survival rates. The disease often presents as a painless neck lump.

**Types of Thyroid Cancer:**
- **Papillary thyroid carcinoma (PTC)**: The most common type (80% of cases), usually slow-growing with excellent prognosis
- **Follicular thyroid carcinoma**: Second most common, generally slow-growing
- **Medullary thyroid carcinoma (MTC)**: Arises from parafollicular C cells, can be hereditary
- **Anaplastic thyroid carcinoma**: Rare, aggressive, and fast-growing

**Common Symptoms:**
- Painless lump or nodule in the neck
- Hoarseness or voice changes
- Difficulty swallowing
- Neck or throat pain
- Swollen lymph nodes

**Risk Factors:**
- Radiation exposure, especially during childhood
- Family history of thyroid cancer or genetic syndromes
- Female gender
- Age 30-60 years

**Diagnosis and Treatment:**
Diagnosis typically involves thyroid ultrasound and fine needle aspiration biopsy. Treatment usually includes surgical removal (thyroidectomy), often followed by radioactive iodine therapy for certain types. The overall prognosis is excellent, with 5-year survival rates exceeding 98% for most types.

=== EXAMPLE 2: COMPLICATIONS QUESTION ===

Question: "What are the complications of total thyroidectomy?"

Answer:
**AI Overview**
Total thyroidectomy, while generally safe, carries several potential complications. The most common include temporary or permanent hypoparathyroidism (affecting calcium levels), recurrent laryngeal nerve injury (causing voice changes), and bleeding. Complication rates vary by surgeon experience and patient factors, with most issues being temporary.

**Major Complications:**
- **Hypoparathyroidism**: Temporary in 20-30% of patients, permanent in 1-3%; causes low calcium levels requiring supplementation
- **Recurrent laryngeal nerve injury**: Temporary in 5-10%, permanent in 1-2%; leads to hoarseness and vocal cord paralysis
- **Bleeding/hematoma**: Occurs in 1-2% of cases, may require emergency surgery if compresses airway
- **Hypothyroidism**: Expected outcome requiring lifelong thyroid hormone replacement

**Less Common Complications:**
- Superior laryngeal nerve injury (affects voice pitch)
- Wound infection (1-2%)
- Seroma formation
- Keloid scarring

**Risk Factors:**
- Extent of surgery (total vs. partial thyroidectomy)
- Surgeon experience and volume
- Revision surgery (higher risk than primary surgery)
- Extent of lymph node dissection

**Prevention and Management:**
Most complications are managed medically. Calcium and vitamin D supplementation prevents symptomatic hypocalcemia. Voice therapy helps with nerve injury recovery. Lifelong thyroid hormone replacement is standard after total thyroidectomy.

=== EXAMPLE 3: PROCEDURE OPTIONS QUESTION ===

Question: "What are the surgical options for differentiated thyroid cancer?"

Answer:
**AI Overview**
Surgical options for differentiated thyroid cancer include lobectomy (removing one thyroid lobe) or total thyroidectomy (removing the entire gland). The choice depends on tumor size, aggressiveness, and patient factors. For small, low-risk tumors (<1-2 cm), lobectomy may suffice. Larger tumors (>4 cm) or those with high-risk features typically require total thyroidectomy.

**Standard Surgical Options:**
- **Lobectomy (hemithyroidectomy)**: Removal of one thyroid lobe and isthmus; suitable for small (<1 cm), low-risk, unifocal tumors with no extrathyroidal extension
- **Total thyroidectomy**: Removal of entire thyroid gland; recommended for tumors >4 cm, bilateral disease, extrathyroidal extension, or lymph node metastases
- **Near-total thyroidectomy**: Removes most thyroid tissue while preserving small amount near parathyroids/recurrent laryngeal nerve

**Lymph Node Management:**
- **Central neck dissection**: Removal of lymph nodes between carotid arteries; performed if nodes are clinically involved
- **Lateral neck dissection**: Removal of side neck nodes if imaging shows metastases

**Factors Influencing Surgical Choice:**
- Tumor size: <1 cm may allow lobectomy; >4 cm requires total thyroidectomy
- Extent of disease: Bilateral, multifocal, or extrathyroidal extension favors total thyroidectomy
- Lymph node status: Positive nodes increase extent of surgery
- Patient age and comorbidities
- Patient preference and desire to avoid lifelong medication

**Newer Approaches:**
- Minimally invasive thyroidectomy (smaller incisions)
- Robotic-assisted surgery (transaxillary or transoral approaches) for cosmetic benefit

=== EXAMPLE 4: COMPARISON QUESTION ===

Question: "What's the difference between lobectomy and total thyroidectomy?"

Answer:
**AI Overview**
Lobectomy removes one thyroid lobe while total thyroidectomy removes the entire gland. The choice depends on cancer size, spread, and patient factors. Lobectomy is less invasive with fewer complications but limits radioactive iodine treatment options. Total thyroidectomy allows for comprehensive cancer treatment but requires lifelong hormone replacement.

**Key Differences:**

| Aspect | Lobectomy | Total Thyroidectomy |
|--------|-----------|---------------------|
| Extent | One lobe + isthmus | Entire gland |
| Best for | Small (<2 cm), low-risk, unifocal tumors | Larger tumors, bilateral disease, high-risk features |
| Hormone replacement | Often not needed (50% of patients) | Always required (lifelong) |
| Complication risk | Lower (1-2% permanent complications) | Higher (2-5% permanent complications) |
| Follow-up | Ultrasound surveillance | Thyroglobulin monitoring + RAI option |
| Reoperation risk | 5-10% may need completion thyroidectomy | None |

**Advantages of Lobectomy:**
- Preserves thyroid function in ~50% of patients
- Lower complication rates
- No lifelong medication for many patients
- Shorter surgery time

**Advantages of Total Thyroidectomy:**
- More complete cancer removal
- Enables radioactive iodine treatment
- Better long-term monitoring with thyroglobulin
- No risk of needing second surgery
- Lower recurrence rates for larger tumors

**When Each is Recommended:**
Lobectomy is appropriate for tumors <2 cm with no concerning features, no lymph node involvement, and no family history. Total thyroidectomy is recommended for tumors >4 cm, bilateral disease, lymph node metastases, or aggressive variants.

=== EXAMPLE 5: DIAGNOSTIC QUESTION ===

Question: "How is papillary thyroid cancer diagnosed?"

Answer:
**AI Overview**
Papillary thyroid cancer is diagnosed through a combination of thyroid ultrasound and fine needle aspiration (FNA) biopsy. The process typically begins when a thyroid nodule is discovered during physical exam or imaging. Ultrasound evaluates suspicious features, and FNA provides cellular diagnosis. Molecular testing may be added for indeterminate results.

**Diagnostic Pathway:**

**1. Initial Detection:**
- Physical examination (palpable neck mass)
- Incidental finding on imaging (CT, MRI, PET scan)
- Thyroid function tests are typically normal

**2. Thyroid Ultrasound:**
- Evaluates nodule characteristics: size, composition, margins, echogenicity
- TI-RADS scoring system classifies nodules by suspicion level
- Suspicious features: hypoechoic, irregular margins, microcalcifications, taller-than-wide shape
- Assesses lymph nodes for metastases

**3. Fine Needle Aspiration (FNA) Biopsy:**
- Performed on nodules ≥1 cm with suspicious features or smaller if high-risk
- Ultrasound-guided for accuracy
- Results reported using Bethesda classification system (I-VI)
- Bethesda VI = malignant (papillary thyroid carcinoma diagnosed)

**4. Molecular Testing:**
- Used for indeterminate FNA results (Bethesda III-IV)
- Tests for BRAF, RAS, RET/PTC mutations
- Helps determine if surgery is needed
- ThyroSeq, Afirma gene expression classifier

**5. Additional Imaging:**
- Neck CT or MRI if extensive disease suspected
- Chest imaging if lung metastases possible
- PET scan rarely needed

**Diagnostic Accuracy:**
FNA has 95% sensitivity and specificity when performed by experienced operators. Molecular testing further improves accuracy for indeterminate cases.

=== EXAMPLE 6: TREATMENT QUESTION ===

Question: "How is medullary thyroid cancer treated?"

Answer:
**AI Overview**
Medullary thyroid cancer (MTC) is treated primarily with surgery, as it doesn't respond to radioactive iodine. Treatment involves total thyroidectomy with central neck lymph node dissection. Genetic testing for RET mutations is essential since 25% of cases are hereditary. Advanced disease may require targeted therapy with tyrosine kinase inhibitors.

**Primary Treatment:**
- **Total thyroidectomy**: Always required; MTC is often bilateral
- **Central neck dissection**: Routinely performed even without obvious lymph node involvement
- **Lateral neck dissection**: If imaging shows lateral compartment metastases

**Genetic Evaluation:**
- RET proto-oncogene testing in all MTC patients
- Identifies hereditary MTC (MEN 2A, MEN 2B, familial MTC)
- Family screening if hereditary form detected
- Prophylactic thyroidectomy for at-risk family members

**Post-Surgical Monitoring:**
- Calcitonin and CEA tumor markers
- Doubling time predicts prognosis
- Imaging (ultrasound, CT, MRI) for structural disease
- No role for radioactive iodine (MTC cells don't take up iodine)

**Advanced Disease Treatment:**
- **Vandetanib**: FDA-approved for progressive, symptomatic, or metastatic MTC
- **Cabozantinib**: Another approved tyrosine kinase inhibitor
- External beam radiation for unresectable local disease
- Liver-directed therapies for liver metastases

**Prognosis:**
10-year survival is 75-85% for localized disease, 40% for lymph node metastases, and 20% for distant metastases. Calcitonin doubling time <2 years indicates aggressive disease.

=== END OF EXAMPLES ===
"""

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
        preview_lines = 15
        
        logger.info("=== CONTEXT PREVIEW (First 15 lines) ===")
        for line in lines[:preview_lines]:
            logger.info(line)
        logger.info("...")
        logger.info(f"=== Total context lines: {len(lines)} ===")

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
        """
        Create adaptive prompt with few-shot examples teaching the LLM different formats.
        """
        few_shot_examples = self._get_few_shot_examples()
        
        return f"""
{self.instructions}

You are a medical information assistant specialized in thyroid cancer. Your task is to provide answers in the style of Google's AI Overview, adapting your format to match the type of question being asked.

=== LEARNING FROM EXAMPLES ===

Study these examples carefully. Notice how the format adapts to the question type:
- Definition questions get overview + types + symptoms + diagnosis
- Complications questions focus on adverse effects + risk factors + prevention
- Procedure questions list options + selection criteria + outcomes
- Comparison questions use side-by-side analysis
- Diagnostic questions follow the diagnostic pathway
- Treatment questions cover primary treatment + monitoring + advanced options

{few_shot_examples}

=== YOUR TASK ===

Now answer the user's question using ONLY the information in the excerpts below. Follow these principles:

**Critical Rules:**
1. **Adapt your format** to match the question type (like the examples above)
2. **Use ONLY information from the excerpts** - never add external knowledge
3. **Extract everything relevant** - if mentioned even once, include it
4. **No inline citations** - write naturally without (Author, Year) references
5. **Be comprehensive** - include percentages, statistics, specific details when available
6. **Natural organization** - use section headers that fit the question, not rigid templates

**Format Flexibility:**
- Start with "AI Overview" (1-2 paragraph summary)
- Use **bold headers** that match the question (not generic labels)
- Organize content logically for THIS specific question
- Include tables for comparisons if helpful
- Omit sections that aren't relevant to this question

**Extraction Guidelines:**
- Read ALL excerpts carefully before writing
- Look for information in results, references, case reports, and discussions
- Include specific data: percentages, frequencies, measurements
- Combine information from multiple sources
- If data is limited, say what you found (not what's missing)

QUESTION: {question}

CONTEXT FROM MEDICAL LITERATURE:
{context}

Now provide your answer, adapting the format to best suit this question:
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
        
        # Log context preview for debugging (comment out in production if needed)
        # self._log_context_preview(context)
        
        confidence = self._compute_confidence(unique_retrieved)

        # Step 5: Generate answer using few-shot learning
        logger.info("Generating final answer with few-shot learning...")
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
