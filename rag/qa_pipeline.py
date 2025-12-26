# rag/qa_pipeline.py
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)

# ---- Minimal intent routing ----
META_PHRASES = [
    "tell me about yourself",
    "who are you",
    "what are you",
    "what is this",
    "what is this ai",
    "what is this app",
    "how does this work",
    "what can you do",
    "help",
]

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

# Evidence mode triggers
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

# User length preferences
SHORT_PREF_PHRASES = ["short", "brief", "concise", "tl;dr", "tldr"]
LONG_PREF_PHRASES = ["detailed", "in detail", "deep", "comprehensive", "everything", "full explanation", "elaborate"]

# Thyroid-only scope keywords
THYROID_KEYWORDS = [
    "thyroid", "thyroid cancer", "thyroid carcinoma",
    "papillary", "ptc",
    "follicular", "ftc",
    "medullary", "mtc",
    "anaplastic", "atc",
    "thyroid nodule", "nodule",
    "tirads", "ti-rads", "tbsrtc",
    "ultrasound", "sono", "echogenic", "microcalcification",
    "biopsy", "fine needle", "fna",
    "radioiodine", "rai", "131i", "iodine ablation", "ablation",
    "thyroglobulin", "tsh", "levothyroxine",
    "neck dissection", "thyroidectomy", "lobectomy",
    "metastasis", "lymph node",
    "braf", "ras", "ret", "molecular",
]


def _norm(text: str) -> str:
    """Lowercase + strip punctuation to make intent matching robust."""
    t = (text or "").strip().lower()
    t = re.sub(r"[\s]+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
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

    # ------------------ Utilities ------------------

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
        parts: List[str] = []
        if self.agent_instructions:
            parts.append(self.agent_instructions)
        if self.rag_instructions:
            parts.append(self.rag_instructions)
        return "\n\n".join(parts).strip()

    def _is_meta_question(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in META_PHRASES)

    def _is_paper_request(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in PAPER_PHRASES)

    def _is_in_scope(self, q: str) -> bool:
        qn = _norm(q)
        return any(_norm(k) in qn for k in THYROID_KEYWORDS)

    def _wants_evidence(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in EVIDENCE_PHRASES)

    def _wants_short(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in SHORT_PREF_PHRASES)

    def _wants_long(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in LONG_PREF_PHRASES)

    def _is_definition_question(self, q: str) -> bool:
        """
        Treat 'tell me about ...' and 'overview of ...' as overview/definition intent too,
        so retrieval becomes more reliable.
        """
        qn = _norm(q)
        return (
            qn.startswith("what is")
            or qn.startswith("define")
            or qn.startswith("explain")
            or qn.startswith("tell me about")
            or qn.startswith("overview of")
            or qn.startswith("describe")
        )

    def _select_mode(self, q: str) -> str:
        """
        Modes:
          - evidence: include verbatim quotes
          - standard: longer structured answer
          - short: shorter answer for simple questions (default for short prompts)
        """
        if self._wants_evidence(q):
            return "evidence"
        if self._wants_long(q):
            return "standard"
        if self._wants_short(q):
            return "short"

        qn = _norm(q)
        wc = len(qn.split())
        if self._is_definition_question(q) and wc <= 10:
            return "short"
        if wc <= 6:
            return "short"
        return "standard"

    def _extract_term(self, q: str) -> str:
        qn = q.strip()
        qn = re.sub(r"[?!.]+$", "", qn).strip()
        ql = qn.lower()
        for prefix in ("what is", "define", "explain", "tell me about", "overview of", "describe"):
            if ql.startswith(prefix):
                return qn[len(prefix):].strip()
        return qn.strip()

    def _extract_pmid(self, q: str) -> Optional[int]:
        m = re.search(r"\bpmid\b[:\s]*([0-9]{6,10})\b", (q or "").lower())
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
            return (
                "That paper is not available in the indexed dataset. "
                "I can only show papers that appear in the retrieved sources."
            )

        if requested_pmid is not None:
            filtered = [r for r in retrieved if r.get("pmid") == requested_pmid]
            if filtered:
                retrieved = filtered

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in retrieved:
            key = str(r.get("pmid") or r.get("title") or "unknown")
            groups.setdefault(key, []).append(r)

        best_key = max(groups.keys(), key=lambda k: len(groups[k]))
        chunks = groups[best_key]
        meta = chunks[0]

        title = (meta.get("title") or "").strip() or "Unknown title"
        year = meta.get("year")
        pmid = meta.get("pmid")
        doi = meta.get("doi")

        lines: List[str] = []
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

        max_excerpts = 8
        shown = 0
        for c in chunks:
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            shown += 1
            lines.append(f"- Excerpt {shown}: {txt}")
            if shown >= max_excerpts:
                break

        if len(chunks) > max_excerpts:
            lines.append(f"\n(Showing {max_excerpts} excerpts out of {len(chunks)} stored for this paper.)")

        return "\n".join(lines)

    # ------------------ Generation helpers ------------------

    def _generate_definition(self, question: str, excerpts: str, mode: str) -> str:
        max_sentences = "1" if mode == "short" else "2"
        prompt_def = f"""
{self.instruction_text}

Retrieved excerpts (use ONLY these):
{excerpts}

User question:
{question}

Task:
Write ONLY the Definition section.

Output format (exactly):
A) Definition
- <{max_sentences} sentence(s)>

Rules:
- Prefer an explicit definition from the excerpts (e.g., “X is …”).
- If there is NO explicit definition, write a *working definition derived only from what the excerpts say*.
  Start with: "Based on the retrieved sources, ..."
- Do NOT add medical facts that are not stated or directly implied by the excerpts.
- End each sentence with citations in the form (Title, Year) if supported.
- If even a working definition is not possible from the excerpts, write exactly:
  A clear definition is not explicitly provided in the retrieved excerpts.
""".strip()
        return self.llm.ask(prompt_def).strip()

    def _extract_facts(self, question: str, excerpts: str, max_facts: int) -> str:
        prompt_extract = f"""
{self.instruction_text}

Retrieved excerpts (use ONLY these):
{excerpts}

User question:
{question}

Task:
Extract ONLY the key facts needed to answer the question.

Output format (exactly):
FACTS:
- <fact sentence> (Title, Year)
- <fact sentence> (Title, Year)

Rules:
- Use ONLY information explicitly stated in the excerpts.
- Each bullet must have a (Title, Year) citation.
- Keep it to at most {max_facts} bullets.
- No interpretation, no recommendations, no extra context.
- If insufficient info, output exactly:
  Not enough evidence in the retrieved sources.
""".strip()
        return self.llm.ask(prompt_extract).strip()

    def _rewrite_summary(self, facts_text: str, min_bullets: int, max_bullets: int) -> str:
        prompt_rewrite = f"""
You are rewriting for clarity and cohesion.

Input facts (do not add anything new):
{facts_text}

Task:
Rewrite the facts into a cohesive set of bullets that are easier to read.
- Keep the same meaning.
- Do NOT introduce any new facts, numbers, study names, or medical recommendations.
- Preserve citations by keeping each citation attached to the bullet that uses that fact.
- Output between {min_bullets} and {max_bullets} bullets.

Output format (exactly):
B) Summary
- <bullet> (Title, Year)
- <bullet> (Title, Year)
""".strip()
        return self.llm.ask(prompt_rewrite).strip()

    def _generate_evidence_quotes(self, question: str, excerpts: str, max_quotes: int) -> str:
        prompt_quotes = f"""
Retrieved excerpts (use ONLY these):
{excerpts}

User question:
{question}

Task:
Provide short direct quotes copied exactly from the excerpts that support the key points.

Rules:
- Quotes must be word-for-word from the excerpts.
- Provide at most {max_quotes} quotes.
- Each quote must be <= 35 words.
- Do not repeat the same idea twice (no duplicate or near-duplicate quotes).
- Prefer quotes that cover different key points.
- After each quote, include (Title, Year, PMID: <PMID>).
- Do not add any text besides the quotes.

Output format (exactly):
C) Verbatim evidence
- "<quote>" (Title, Year, PMID: <PMID>)
""".strip()
        return self.llm.ask(prompt_quotes).strip()

    # ------------------ Main entry ------------------

    def answer(self, question: str, chat_history: Optional[list] = None, k: int = 5) -> str:
        try:
            # Meta questions: answer directly (no retrieval)
            if self._is_meta_question(question):
                return (
                    "I’m the assistant inside **Thyroid Cancer RAG Assistant**.\n\n"
                    "- I answer **thyroid cancer** questions by retrieving relevant excerpts from your indexed dataset (Qdrant).\n"
                    "- Then I generate an evidence-grounded response using only those excerpts.\n\n"
                    "Ask about: thyroid cancer types (PTC/FTC/MTC/ATC), nodules, ultrasound/TIRADS, biopsy, staging, surgery, "
                    "radioiodine (RAI), follow-up, recurrence, and prognosis."
                )

            # Scope: thyroid-only
            if not self._is_in_scope(question):
                return (
                    "I’m scoped to **thyroid cancer** questions only.\n\n"
                    "Try asking about thyroid nodules, ultrasound features, biopsy/FNA, thyroid cancer subtypes, staging, "
                    "surgery, radioiodine (RAI), follow-up, or recurrence."
                )

            # Paper request mode
            if self._is_paper_request(question):
                requested_pmid = self._extract_pmid(question)
                retrieved = self.vector_store.search(question, k=max(k, 10))
                return self._paper_lookup_response(retrieved, requested_pmid)

            # Select answer mode
            mode = self._select_mode(question)  # "short" | "standard" | "evidence"

            # Build retrieval query
            retrieval_query = question
            if self._is_definition_question(question):
                term = self._extract_term(question)
                retrieval_query = f"definition of {term}" if term else retrieval_query

            # Retrieval depth by mode
            if mode == "short":
                k = max(k, 8)
            else:
                k = max(k, 12)

            retrieved = self.vector_store.search(retrieval_query, k=k)

            # Fallback retry for broad/overview queries
            if not retrieved:
                term = self._extract_term(question)
                expanded = f"overview of {term} thyroid cancer" if term else f"overview of {question}"
                retrieved = self.vector_store.search(expanded, k=max(k, 12))

            if not retrieved:
                return "Not enough evidence in the retrieved sources."

            excerpts = self._format_excerpts(retrieved)

            # Generate sections
            definition_section = self._generate_definition(question, excerpts, mode=mode)

            if mode == "short":
                facts = self._extract_facts(question, excerpts, max_facts=4)
                if facts.strip() == "Not enough evidence in the retrieved sources.":
                    return facts.strip()
                summary_section = self._rewrite_summary(facts, min_bullets=2, max_bullets=3)
                final_answer = f"{definition_section}\n\n{summary_section}".strip()
                return final_answer

            # standard/evidence
            facts = self._extract_facts(question, excerpts, max_facts=8)
            if facts.strip() == "Not enough evidence in the retrieved sources.":
                return facts.strip()

            summary_section = self._rewrite_summary(facts, min_bullets=3, max_bullets=6)

            if mode == "evidence":
                evidence_section = self._generate_evidence_quotes(question, excerpts, max_quotes=5)
                final_answer = f"{definition_section}\n\n{summary_section}\n\n{evidence_section}".strip()
            else:
                final_answer = f"{definition_section}\n\n{summary_section}".strip()

            return final_answer

        except Exception as e:
            logging.error(f"Error during answer generation: {e}")
            return "⚠️ Something went wrong while generating the answer. Please try again."
