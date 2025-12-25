import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)

META_KEYWORDS = [
    "what is this", "what is this ai", "who are you", "tell me about yourself",
    "what can you do", "how does this work", "what is this app", "about this"
]

PAPER_KEYWORDS = [
    "give me the paper", "show the paper", "link the paper", "open the paper",
    "full text", "pdf", "download", "paper link", "pubmed", "doi", "pmid"
]

# Keep scope strictly thyroid-cancer-related
THYROID_KEYWORDS = [
    "thyroid", "papillary", "follicular", "medullary", "anaplastic",
    "ptc", "ftc", "mtc", "atc", "thyroid nodule", "tirads", "ultrasound",
    "radioiodine", "rai", "131i", "levothyroxine", "thyroglobulin", "tsh"
]


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
        if env == "prod":
            base_dir = Path(__file__).parent
            rag_path = base_dir / rag_instruction_file
            agent_path = base_dir / agent_instruction_file
        else:
            rag_path = Path(rag_instruction_file)
            agent_path = Path(agent_instruction_file)

        self.rag_instructions = self._load_instruction(rag_path)
        self.agent_instructions = self._load_instruction(agent_path)
        self.instruction_text = self._combine_instructions()

    def _load_instruction(self, file_path: Path) -> str:
        if not file_path.exists():
            logging.warning(f"Instruction file {file_path} not found.")
            return ""
        try:
            return file_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logging.error(f"Failed to load instruction file {file_path}: {e}")
            return ""

    def _combine_instructions(self) -> str:
        parts = []
        if self.agent_instructions:
            parts.append(self.agent_instructions)
        if self.rag_instructions:
            parts.append(self.rag_instructions)
        return "\n\n".join(parts).strip()

    def _format_chat_history(self, chat_history: Optional[list]) -> str:
        if not chat_history:
            return ""
        lines: List[str] = []
        for msg in chat_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    def _is_meta_question(self, q: str) -> bool:
        ql = q.strip().lower()
        return any(k in ql for k in META_KEYWORDS)

    def _is_paper_request(self, q: str) -> bool:
        ql = q.strip().lower()
        return any(k in ql for k in PAPER_KEYWORDS)

    def _is_in_scope(self, q: str) -> bool:
        ql = q.strip().lower()
        return any(k in ql for k in THYROID_KEYWORDS)

    def _is_definition_question(self, q: str) -> bool:
        ql = q.strip().lower()
        return ql.startswith("what is") or ql.startswith("define") or ql.startswith("explain")

    def _extract_pmid(self, q: str) -> Optional[int]:
        # Match "PMID 12345678" or any 6-10 digit number after PMID
        m = re.search(r"\bpmid\b[:\s]*([0-9]{6,10})\b", q.lower())
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _format_excerpts(self, retrieved: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for i, item in enumerate(retrieved, start=1):
            title = (item.get("title") or "").strip()
            year = item.get("year", "")
            pmid = item.get("pmid", "")
            doi = item.get("doi", "")
            evidence = item.get("evidence_level", "")
            text = (item.get("text") or "").strip()

            if title and year:
                source_label = f"{title} ({year})"
            elif title:
                source_label = title
            else:
                source_label = f"Source {i}"

            parts.append(
                f"SOURCE {i}: {source_label}\n"
                f"PMID: {pmid} | DOI: {doi} | Year: {year} | Evidence level: {evidence}\n"
                f"EXCERPT:\n{text}"
            )
        return "\n\n".join(parts).strip()

    def _paper_lookup_response(self, retrieved: List[Dict[str, Any]], requested_pmid: Optional[int]) -> str:
        if not retrieved:
            return "That paper is not available in the indexed dataset. I can only show papers that appear in the retrieved sources."

        # If user mentioned a PMID, filter to it
        if requested_pmid is not None:
            filtered = [r for r in retrieved if r.get("pmid") == requested_pmid]
            if filtered:
                retrieved = filtered

        # Group by PMID or title
        # We'll pick the top paper by frequency (and fall back to first)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in retrieved:
            key = str(r.get("pmid") or r.get("title") or "unknown")
            groups.setdefault(key, []).append(r)

        # Choose group with most chunks
        best_key = max(groups.keys(), key=lambda k: len(groups[k]))
        chunks = groups[best_key]

        # Use first chunk metadata as paper metadata
        meta = chunks[0]
        title = (meta.get("title") or "").strip() or "Unknown title"
        year = meta.get("year")
        pmid = meta.get("pmid")
        doi = meta.get("doi")

        lines = []
        lines.append("Paper details:")
        lines.append(f"- Title: {title}")
        lines.append(f"- Year: {year if year else 'Not available'}")
        lines.append(f"- PMID: {pmid if pmid else 'Not available'}")
        lines.append(f"- DOI: {doi if doi else 'DOI not available in the indexed metadata.'}")

        if pmid:
            lines.append(f"- PubMed link: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        if doi:
            lines.append(f"- DOI link: https://doi.org/{doi}")

        lines.append("")
        lines.append("Available excerpts in this database:")

        # Show up to 8 excerpts to keep UI readable
        max_excerpts = 8
        for i, c in enumerate(chunks[:max_excerpts], start=1):
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            lines.append(f"- Excerpt {i}: {txt}")

        if len(chunks) > max_excerpts:
            lines.append(f"\n(Showing {max_excerpts} excerpts out of {len(chunks)} stored for this paper.)")

        return "\n".join(lines)

    def answer(self, question: str, chat_history: Optional[list] = None, k: int = 5) -> str:
        try:
            # 1) Meta questions about the app/AI
            if self._is_meta_question(question):
                return (
                    "This is **Thyroid Cancer RAG Assistant** — a research-focused chatbot that answers *thyroid cancer* questions.\n\n"
                    "- It retrieves relevant excerpts from an indexed dataset (Qdrant vector database)\n"
                    "- Then generates an answer grounded in those excerpts (with citations)\n\n"
                    "Scope: **thyroid cancer only** (e.g., types, diagnosis, ultrasound, pathology, staging, treatment, outcomes).\n"
                    "If you ask something outside this scope, I’ll ask you to rephrase toward thyroid cancer."
                )

            # 2) Enforce scope (thyroid cancer only)
            if not self._is_in_scope(question):
                return (
                    "I’m scoped to **thyroid cancer** questions only.\n\n"
                    "Try asking about thyroid cancer types (PTC/FTC/MTC/ATC), nodules, ultrasound features, biopsy, staging, "
                    "treatment (surgery/RAI), follow-up, or prognosis."
                )

            chat_context = self._format_chat_history(chat_history)

            # 3) Paper lookup mode
            if self._is_paper_request(question):
                requested_pmid = self._extract_pmid(question)
                retrieved = self.vector_store.search(question, k=max(k, 10))
                return self._paper_lookup_response(retrieved, requested_pmid)

            # 4) Normal thyroid-cancer RAG mode
            retrieval_query = question
            if self._is_definition_question(question):
                k = max(k, 12)
                retrieval_query = f"definition of {question}"

            retrieved = self.vector_store.search(retrieval_query, k=k)
            if not retrieved:
                return "Not enough evidence in the retrieved sources."

            excerpts = self._format_excerpts(retrieved)

            prompt = f"""
{self.instruction_text}

Conversation history (may be empty):
{chat_context}

Retrieved excerpts (use ONLY these):
{excerpts}

User question:
{question}

OUTPUT FORMAT (follow exactly):

A) Definition (1–2 sentences)
- If the retrieved excerpts contain an explicit definition (e.g., “X is …”), write it as a short definition.
- If NO explicit definition is present, write exactly:
  A clear definition is not explicitly provided in the retrieved excerpts.
- End this section with a citation like (Title, Year) ONLY if supported.

B) Summary (paraphrase, 3–6 bullet points)
- Summarize ONLY what the excerpts support.
- After EACH bullet, cite using (Title, Year).

C) Verbatim evidence (word-for-word quotes)
- Provide 3–6 short direct quotes copied exactly from the excerpts.
- Each quote must be ≤ 35 words.
- After each quote, attribute with (Title, Year, PMID).

Rules:
- Use ONLY the retrieved excerpts. Do not use outside knowledge.
- If the excerpts do not support any meaningful answer, respond exactly:
  Not enough evidence in the retrieved sources.

Answer:
""".strip()

            response = self.llm.ask(prompt)

            # If model slips format once, force rewrite (keeps UX consistent)
            if "A) Definition" not in response:
                fix_prompt = f"""
Rewrite the answer to follow EXACTLY this format:

A) Definition (1–2 sentences)
B) Summary (3–6 bullet points)
C) Verbatim evidence (3–6 short quotes, ≤35 words each)

Use ONLY the retrieved excerpts already provided. Do not add new facts.
Citations: (Title, Year) in A and B; (Title, Year, PMID) in C.

Original answer:
{response}
"""
                response = self.llm.ask(fix_prompt)

            return response

        except Exception as e:
            logging.error(f"Error during answer generation: {e}")
            return "⚠️ Something went wrong while generating the answer. Please try again."
