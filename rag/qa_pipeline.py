# rag/qa_pipeline.py
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)

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
MAX_SOURCES = 6
MAX_CHUNKS_PER_SOURCE = 2
MAX_EXCERPT_CHARS = 900
MAX_TOTAL_CONTEXT_CHARS = 6500


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
        """Create Google-style overview prompt."""
        return f"""
{self.instructions}

You are a medical information assistant specialized in thyroid cancer. Your task is to provide clear, patient-friendly answers in the style of Google's AI Overview.

CRITICAL RULES:
1. Use ONLY information from the provided excerpts
2. Do NOT cite sources inline (no parenthetical citations)
3. Do NOT mention confidence scores or evidence levels in the main answer
4. Do NOT add information not present in the excerpts
5. Write in clear, accessible language for patients

OUTPUT FORMAT (follow this structure exactly):

**AI Overview**
[Write 1-2 paragraph direct answer summarizing the key information]

**[Main Topic Category - e.g., "Standard Surgical Options"]:**
- **Option 1**: [Clear description]
- **Option 2**: [Clear description]
- **Option 3**: [Clear description]

**Factors Influencing [Decision/Choice]:**
- **Factor 1**: [Explanation]
- **Factor 2**: [Explanation]
- **Factor 3**: [Explanation]

**Alternative/Additional Considerations:**
- [Point 1]
- [Point 2]

**Potential Risks:**
- [Risk 1]
- [Risk 2]

Note: Adapt the section headers based on what's relevant to the question. Not all sections are always needed.

QUESTION: {question}

CONTEXT FROM MEDICAL LITERATURE:
{context}

Now provide your answer following the format above:
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
        k: int = 25
    ) -> str:
        """
        Generate a Google-style overview answer to the question.
        
        Args:
            question: User's question
            chat_history: Optional conversation history (not currently used)
            k: Number of chunks to retrieve
            
        Returns:
            Formatted answer with sources and evidence quality
        """
        # Retrieve relevant chunks
        retrieved = self.vector_store.search(question, k=k)
        
        if not retrieved:
            return "I don't have enough information in my knowledge base to answer this question about thyroid cancer."

        # Build context and compute confidence
        context = self._build_context(retrieved)
        confidence = self._compute_confidence(retrieved)

        # Generate answer
        prompt = self._create_prompt(question, context)
        answer = self.llm.ask(prompt).strip()

        # Add evidence summary and sources
        sources = self._extract_sources(retrieved)
        
        result = f"""{answer}

---

**Evidence Quality:** {confidence['label']} confidence ({confidence['score']}/100)
*Based on: {confidence['breakdown']}*

**Key Sources:**
{chr(10).join(sources) if sources else "• No sources available"}
""".strip()

        return result
