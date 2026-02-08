# rag/qa_pipeline.py
import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

logging.basicConfig(level=logging.INFO)

PAPER_PHRASES = [
    "give me the paper",
    "show the paper",
    "link the paper",
    "open the paper",
    "get the paper",
    "send the paper",
    "full text",
    "pdf",
    "download",
    "paper link",
    "pubmed",
    "doi",
    "pmid",
]

EVIDENCE_PHRASES = [
    "show evidence",
    "show quotes",
    "quotes",
    "verbatim",
    "excerpts",
    "show excerpts",
    "cite",
    "citations",
    "sources",
    "show sources",
    "proof",
]

SHORT_PREF_PHRASES = ["short", "brief", "concise", "tl;dr", "tldr"]
LONG_PREF_PHRASES = ["detailed", "in detail", "deep", "comprehensive", "everything", "full explanation", "elaborate"]

SCOPE_ANCHORS = [
    "thyroid",
    "thyroid cancer",
    "thyroid carcinoma",
    "papillary thyroid carcinoma",
    "follicular thyroid carcinoma",
    "medullary thyroid carcinoma",
    "anaplastic thyroid carcinoma",
    "thyroid nodule",
    "tirads",
    "fine needle aspiration",
    "radioiodine",
    "thyroidectomy",
    "braf",
    "ret",
]

EVIDENCE_LEVEL_INFO: Dict[int, Tuple[str, float]] = {
    1: ("Guidelines / Consensus", 1.00),
    2: ("Systematic Review / Meta-analysis", 0.90),
    3: ("Randomized Controlled Trials", 0.80),
    4: ("Clinical Trials (non-randomized)", 0.70),
    5: ("Cohort Studies", 0.60),
    6: ("Case-Control Studies", 0.50),
    7: ("Case Reports / Series", 0.40),
}

MAX_SOURCES = 6
MAX_CHUNKS_PER_SOURCE = 2
MAX_EXCERPT_CHARS = 900
MAX_TOTAL_CONTEXT_CHARS = 6500


def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s\-]", "", t)
    return t


class QAPipeline:
    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        llm_client: Any,
        rag_instruction_file: str = "instructions/rag_instructions.txt",
        agent_instruction_file: str = "instructions/agent_instructions.txt",
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm_client

        env = os.getenv("ENV", "local").lower()
        base = Path(__file__).parent if env == "prod" else Path(".")
        self.rag_instructions = (base / rag_instruction_file).read_text(encoding="utf-8")
        self.agent_instructions = (base / agent_instruction_file).read_text(encoding="utf-8")

        self.instruction_text = f"{self.agent_instructions}\n\n{self.rag_instructions}".strip()

    # -------------------- mode selection --------------------

    def _select_mode(self, q: str) -> str:
        if any(p in _norm(q) for p in EVIDENCE_PHRASES):
            return "evidence"
        if any(p in _norm(q) for p in SHORT_PREF_PHRASES):
            return "short"
        return "overview"

    # -------------------- confidence --------------------

    def _compute_confidence(self, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        levels = [r.get("evidence_level") for r in retrieved if r.get("evidence_level") in EVIDENCE_LEVEL_INFO]
        if not levels:
            return {"label": "Low", "score": 0, "breakdown": "No evidence metadata."}

        weights = [EVIDENCE_LEVEL_INFO[l][1] for l in levels]
        score = int(round((sum(weights) / len(weights)) * 100))
        label = "High" if score >= 85 else "Medium" if score >= 65 else "Low"

        breakdown = "; ".join(
            f"Level {l} ({EVIDENCE_LEVEL_INFO[l][0]}): {levels.count(l)}"
            for l in sorted(set(levels))
        )

        return {"label": label, "score": score, "breakdown": breakdown}

    def _format_confidence_block(self, conf: Dict[str, Any]) -> str:
        return (
            f"- Confidence: **{conf['label']}** ({conf['score']}/100)\n"
            f"- Evidence levels: {conf['breakdown']}"
        )

    # -------------------- context builder --------------------

    def _build_context(self, retrieved: List[Dict[str, Any]]) -> str:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in retrieved:
            key = str(r.get("pmid") or r.get("title"))
            grouped.setdefault(key, []).append(r)

        parts = []
        total = 0

        for idx, group in enumerate(grouped.values(), start=1):
            meta = group[0]
            header = (
                f"SOURCE {idx}: {meta.get('title')} ({meta.get('year')})\n"
                f"PMID: {meta.get('pmid')} | Evidence level: {meta.get('evidence_level')}\n"
            )
            if total + len(header) > MAX_TOTAL_CONTEXT_CHARS:
                break
            parts.append(header)
            total += len(header)

            for g in group[:MAX_CHUNKS_PER_SOURCE]:
                text = g.get("text", "")[:MAX_EXCERPT_CHARS]
                block = f"EXCERPT:\n{text}\n"
                if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
                    break
                parts.append(block)
                total += len(block)

        return "\n".join(parts)

    # -------------------- prompts --------------------

    def _prompt_overview(self, question: str, context: str) -> str:
        return f"""
{self.instruction_text}

Task:
Write a clear, patient-friendly medical overview similar to Google's AI Overview.

Rules:
- Use ONLY the information in the excerpts
- Do NOT cite sources inline
- Do NOT mention confidence scores or evidence levels
- Do NOT add information not present in the excerpts

OUTPUT FORMAT (follow exactly):

AI Overview
<1–2 paragraph summary>

Standard Surgical Options
- ...

Factors Influencing Surgical Choice
- ...

Other Management Approaches
- ...

Potential Risks
- ...

Context:
{context}
""".strip()

    def _prompt_normal(self, question: str, context: str) -> str:
        return f"""
{self.instruction_text}

You MUST answer using ONLY the excerpts in the context.

OUTPUT FORMAT:
A) Definition
- 1–2 bullets (Title, Year)

B) Summary
- 3–5 bullets (Title, Year)

Context:
{context}
""".strip()

    # -------------------- answer --------------------

    def answer(self, question: str, chat_history: Optional[list] = None, k: int = 25) -> str:
        mode = self._select_mode(question)

        retrieved = self.vector_store.search(question, k=max(k, 12))
        if not retrieved:
            return "Not enough evidence in the retrieved sources."

        context = self._build_context(retrieved)
        conf = self._compute_confidence(retrieved)
        conf_block = self._format_confidence_block(conf)

        if mode == "overview":
            draft = self.llm.ask(self._prompt_overview(question, context)).strip()

            sources = []
            seen = set()
            for r in retrieved:
                key = (r.get("title"), r.get("year"))
                if key not in seen:
                    seen.add(key)
                    sources.append(f"- {r.get('title')} ({r.get('year')})")
                if len(sources) >= MAX_SOURCES:
                    break

            return f"""{draft}

---
Evidence Summary (optional)
{conf_block}

Key Sources
{chr(10).join(sources)}
""".strip()

        return self.llm.ask(self._prompt_normal(question, context)).strip()
