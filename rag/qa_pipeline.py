# rag/qa_pipeline.py
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

logging.basicConfig(level=logging.INFO)

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

# ✅ NEW: credibility / fact-check intent
CREDIBILITY_PHRASES = [
    "credibility",
    "credible",
    "is this true",
    "is this correct",
    "fact check",
    "fact-check",
    "verify",
    "validate",
    "check this claim",
    "check this",
    "is this accurate",
    "how accurate is this",
    "cross check",
    "cross-check",
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

        self._scope_anchors_norm = sorted({_norm(a) for a in SCOPE_ANCHORS if a})

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
        parts: List[str] = []
        if self.agent_instructions:
            parts.append(self.agent_instructions)
        if self.rag_instructions:
            parts.append(self.rag_instructions)
        return "\n\n".join(parts).strip()

    # ---------------- routing helpers ----------------

    def _is_meta_question(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in META_PHRASES)

    def _is_paper_request(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in PAPER_PHRASES)

    def _wants_evidence(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in EVIDENCE_PHRASES)

    def _wants_short(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in SHORT_PREF_PHRASES)

    def _wants_long(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in LONG_PREF_PHRASES)

    def _is_credibility_check(self, q: str) -> bool:
        qn = _norm(q)
        return any(p in qn for p in CREDIBILITY_PHRASES)

    def _is_definition_question(self, q: str) -> bool:
        qn = _norm(q)
        return (
            qn.startswith("what is")
            or qn.startswith("define")
            or qn.startswith("explain")
            or qn.startswith("tell me about")
            or qn.startswith("overview of")
            or qn.startswith("describe")
        )

    def _extract_term(self, q: str) -> str:
        raw = (q or "").strip()
        raw = re.sub(r"[?!.]+$", "", raw).strip()
        ql = raw.lower()
        for prefix in ("what is", "define", "explain", "tell me about", "overview of", "describe"):
            if ql.startswith(prefix):
                return raw[len(prefix):].strip()
        return raw

    def _extract_claim_text(self, q: str) -> str:
        """
        Try to strip credibility phrasing and keep the user-provided claim.
        Works even if user writes: "Is this true: <claim>" or "Check credibility of <claim>"
        """
        s = (q or "").strip()

        # Remove common wrappers
        s = re.sub(r"(?i)^\s*(is this true|is this correct|is this accurate|fact[- ]?check|verify|validate|check credibility of|credibility check)\s*[:\-]?\s*", "", s)
        s = re.sub(r"(?i)^\s*(check this claim|check this|cross[- ]?check)\s*[:\-]?\s*", "", s)

        # Remove trailing question marks
        s = s.strip()
        s = re.sub(r"[?]+$", "", s).strip()

        return s if s else (q or "").strip()

    def _select_mode(self, q: str) -> str:
        if self._wants_evidence(q):
            return "evidence"
        if self._wants_long(q):
            return "standard"
        if self._wants_short(q):
            return "short"

        wc = len(_norm(q).split())
        if self._is_definition_question(q) and wc <= 10:
            return "short"
        if wc <= 6:
            return "short"
        return "standard"

    # ---------------- scope check ----------------

    def _is_in_scope(self, q: str) -> bool:
        qn = _norm(q)
        if not qn:
            return False
        if "thyroid" in qn:
            return True
        for tok in qn.split():
            if fuzz.ratio(tok, "thyroid") >= 80:
                return True
        match = process.extractOne(qn, self._scope_anchors_norm, scorer=fuzz.token_set_ratio)
        return bool(match and match[1] >= 75)

    # ---------------- level filter parsing ----------------

    def _parse_level_filter(self, q: str) -> Optional[List[int]]:
        qn = _norm(q)
        if "level" not in qn and "levels" not in qn:
            return None

        m_range = re.search(r"\blevels?\s*([1-7])\s*-\s*([1-7])\b", qn)
        if m_range:
            a, b = int(m_range.group(1)), int(m_range.group(2))
            lo, hi = min(a, b), max(a, b)
            return list(range(lo, hi + 1))

        nums = re.findall(r"\blevels?\s*[:\-]?\s*([1-7])\b", qn)
        levels = sorted({int(n) for n in nums}) if nums else []

        if not levels:
            m_from = re.search(r"\bfrom\s+level\s+([1-7])\b", qn)
            if m_from:
                levels = [int(m_from.group(1))]

        return levels or None

    def _strip_level_filter_text(self, q: str) -> str:
        s = q or ""
        s = re.sub(r"\bonly\s+from\s+levels?\s*[1-7](\s*-\s*[1-7])?\b", "", s, flags=re.I)
        s = re.sub(r"\bfrom\s+levels?\s*[1-7](\s*-\s*[1-7])?\b", "", s, flags=re.I)
        s = re.sub(r"\blevels?\s*[1-7](\s*-\s*[1-7])?\s*only\b", "", s, flags=re.I)
        s = re.sub(r"\blevels?\s*[1-7](\s*(and|,)\s*[1-7])+\b", "", s, flags=re.I)
        s = re.sub(r"\blevel\s*[1-7]\b", "", s, flags=re.I)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ---------------- confidence rating ----------------

    def _safe_int(self, x: Any) -> Optional[int]:
        try:
            if x is None or isinstance(x, bool):
                return None
            return int(x)
        except Exception:
            return None

    def _compute_confidence(self, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_source: Dict[str, Dict[str, Any]] = {}
        for r in retrieved or []:
            pmid = r.get("pmid")
            key = str(pmid) if pmid else str(r.get("title") or id(r))
            if key not in by_source:
                by_source[key] = r

        levels: List[int] = []
        for src in by_source.values():
            lvl = self._safe_int(src.get("evidence_level"))
            if lvl in EVIDENCE_LEVEL_INFO:
                levels.append(lvl)

        if not levels:
            return {"label": "Low", "score": 0, "breakdown": "No evidence level metadata found."}

        weights = [EVIDENCE_LEVEL_INFO[l][1] for l in levels]
        avg_weight = sum(weights) / len(weights)

        n = len(levels)
        if n == 1:
            avg_weight *= 0.90
        elif n == 2:
            avg_weight *= 0.95

        score = int(round(avg_weight * 100))
        if score >= 85:
            label = "High"
        elif score >= 65:
            label = "Medium"
        else:
            label = "Low"

        counts: Dict[int, int] = {}
        for l in levels:
            counts[l] = counts.get(l, 0) + 1

        parts = [f"Level {lvl} ({EVIDENCE_LEVEL_INFO[lvl][0]}): {counts[lvl]}" for lvl in sorted(counts)]
        breakdown = "; ".join(parts)

        return {"label": label, "score": score, "breakdown": breakdown}

    def _format_confidence_block(self, conf: Dict[str, Any]) -> str:
        return (
            "Confidence rating\n"
            f"- **{conf['label']}** ({conf['score']}/100)\n"
            f"- Based on retrieved evidence levels: {conf['breakdown']}"
        )

    # ---------------- paper mode ----------------

    def _extract_pmid(self, q: str) -> Optional[int]:
        m = re.search(r"\bpmid\b[:\s]*([0-9]{6,10})\b", (q or "").lower())
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

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

        shown = 0
        for c in chunks:
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            shown += 1
            lines.append(f"- Excerpt {shown}: {txt[:700].rstrip()}…")
            if shown >= 6:
                break

        return "\n".join(lines)

    # ---------------- context builder ----------------

    def _build_context(self, retrieved: List[Dict[str, Any]]) -> str:
        def _score(x: Dict[str, Any]) -> float:
            s = x.get("score")
            try:
                return float(s)
            except Exception:
                return -1.0

        ranked = sorted(retrieved or [], key=_score, reverse=True)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in ranked:
            pmid = r.get("pmid")
            key = str(pmid) if pmid else str(r.get("title") or "unknown")
            grouped.setdefault(key, []).append(r)

        source_keys = sorted(grouped.keys(), key=lambda k: _score(grouped[k][0]), reverse=True)[:MAX_SOURCES]

        parts: List[str] = []
        total = 0
        src_idx = 0

        for key in source_keys:
            src_idx += 1
            hits = grouped[key][:MAX_CHUNKS_PER_SOURCE]

            meta = hits[0]
            title = (meta.get("title") or "").strip()
            year = meta.get("year", "")
            pmid = meta.get("pmid", "")
            doi = meta.get("doi", "")
            level = meta.get("evidence_level", "")

            if title and year:
                label = f"{title} ({year})"
            elif title:
                label = title
            else:
                label = f"Source {src_idx}"

            header = f"SOURCE {src_idx}: {label}\nPMID: {pmid} | DOI: {doi} | Year: {year} | Evidence level: {level}\n"
            if total + len(header) > MAX_TOTAL_CONTEXT_CHARS:
                break
            parts.append(header)
            total += len(header)

            for h in hits:
                text = (h.get("text") or "").strip()
                if not text:
                    continue
                if len(text) > MAX_EXCERPT_CHARS:
                    text = text[:MAX_EXCERPT_CHARS].rstrip() + "…"

                block = f"EXCERPT:\n{text}\n"
                if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
                    break
                parts.append(block)
                total += len(block)

            if total >= MAX_TOTAL_CONTEXT_CHARS:
                break

        return "\n".join(parts).strip()

    # ---------------- credibility prompt ----------------

    def _build_credibility_prompt(self, claim: str, context: str) -> str:
        return f"""
You are verifying the credibility of a user-provided claim using ONLY the excerpts provided.

CRITICAL RULES:
- Use ONLY the excerpts in the context. Do NOT use outside knowledge or the web.
- If the excerpts do not address the claim, say "Insufficient evidence in the retrieved sources."
- Be strict: "Supported" only if the sources clearly support the claim.
- If sources contradict the claim, label as "Contradicted".
- If only some parts are supported, label as "Partially supported".
- Every bullet must end with citations in the form (Title, Year).
- Quotes must be word-for-word from excerpts and <= 35 words.

OUTPUT FORMAT (follow exactly):

Credibility check
- Claim: <repeat the claim>

- Verdict: Supported / Partially supported / Not supported / Contradicted / Insufficient evidence

- Rationale:
- <bullet> (Title, Year)
- <bullet> (Title, Year)
(2–4 bullets)

- Supporting evidence (quotes):
- "<quote>" (Title, Year, PMID: <PMID>)
- "<quote>" (Title, Year, PMID: <PMID>)
(0–3 quotes; include only if available)

Claim:
{claim}

Context (excerpts):
{context}
""".strip()

    # ---------------- insert confidence ----------------

    def _insert_confidence_after_header(self, draft: str, conf_block: str) -> str:
        # Put confidence right after "Credibility check" header if present
        if "Credibility check" in draft:
            parts = draft.split("Credibility check", 1)
            return "Credibility check\n\n" + conf_block + "\n\n" + parts[1].lstrip()
        return f"{draft}\n\n{conf_block}".strip()

    # ---------------- main API ----------------

    def answer(self, question: str, chat_history: Optional[list] = None, k: int = 5) -> str:
        try:
            if self._is_meta_question(question):
                return (
                    "I’m the assistant inside **Thyroid Cancer RAG Assistant**.\n\n"
                    "- I answer **thyroid cancer** questions by retrieving relevant excerpts from your indexed dataset (Qdrant).\n"
                    "- Then I generate an evidence-grounded response using only those excerpts.\n\n"
                    "You can ask about: thyroid cancer types (PTC/FTC/MTC/ATC), nodules, ultrasound/TIRADS, biopsy/FNA, "
                    "staging, surgery, radioiodine (RAI), follow-up, recurrence, and prognosis.\n\n"
                    "You can also paste a claim and ask me to **check credibility** against the indexed literature."
                )

            requested_levels = self._parse_level_filter(question)
            q_clean = self._strip_level_filter_text(question)

            if not self._is_in_scope(q_clean):
                return (
                    "I’m scoped to **thyroid cancer** questions only.\n\n"
                    "Try asking about thyroid nodules, ultrasound features, biopsy/FNA, thyroid cancer subtypes, staging, "
                    "surgery, radioiodine (RAI), follow-up, or recurrence."
                )

            if self._is_paper_request(question):
                requested_pmid = self._extract_pmid(question)
                retrieved = self.vector_store.search(q_clean, k=max(k, 12), levels=requested_levels)
                return self._paper_lookup_response(retrieved, requested_pmid)

            # ✅ NEW: credibility check mode
            if self._is_credibility_check(question):
                claim = self._extract_claim_text(question)
                # Retrieve more aggressively for verification
                retrieved = self.vector_store.search(claim, k=max(k, 14), levels=requested_levels)

                if not retrieved:
                    return "Insufficient evidence in the retrieved sources."

                conf = self._compute_confidence(retrieved)
                conf_block = self._format_confidence_block(conf)

                context = self._build_context(retrieved)
                prompt = self._build_credibility_prompt(claim=claim, context=context)

                draft = self.llm.ask(prompt).strip()
                if draft.startswith("⚠️"):
                    return draft

                return self._insert_confidence_after_header(draft, conf_block)

            # Normal QA mode
            mode = self._select_mode(q_clean)

            retrieval_query = q_clean
            if self._is_definition_question(q_clean):
                term = self._extract_term(q_clean)
                if term:
                    retrieval_query = f"definition of {term}"

            k_use = max(k, 8) if mode == "short" else max(k, 12)
            if requested_levels == [1]:
                k_use = min(k_use, 6)

            retrieved = self.vector_store.search(retrieval_query, k=k_use, levels=requested_levels)

            if not retrieved:
                term = self._extract_term(q_clean)
                expanded = f"overview of {term} thyroid cancer" if term else f"overview of {q_clean}"
                retrieved = self.vector_store.search(expanded, k=max(k_use, 10), levels=requested_levels)

            if not retrieved:
                if requested_levels:
                    return (
                        f"Not enough evidence in the retrieved sources for **Level(s) {requested_levels}**.\n\n"
                        "Try removing the level filter (or asking a narrower question)."
                    )
                return "Not enough evidence in the retrieved sources."

            conf = self._compute_confidence(retrieved)
            conf_block = self._format_confidence_block(conf)

            context = self._build_context(retrieved)

            # Simple QA prompt
            if mode == "short":
                fmt = "A) Answer\n- 2–3 bullets.\n\nB) Key citations\n- 2–4 bullets."
            elif mode == "evidence":
                fmt = "A) Answer\n- 3–6 bullets.\n\nB) Verbatim evidence\n- 3–5 quotes (<=35 words)."
            else:
                fmt = "A) Answer\n- 3–6 bullets.\n\nB) Key citations\n- 2–6 bullets."

            prompt = f"""
You are a cautious clinical assistant answering thyroid cancer questions using ONLY the provided excerpts.

CRITICAL RULES:
- Use ONLY the excerpts in the context. Do NOT add outside medical knowledge.
- If the excerpts do not support an answer, say exactly: "Not enough evidence in the retrieved sources."
- Every bullet must end with citations in the form (Title, Year).
- Do not invent citations.

OUTPUT FORMAT (follow exactly):
{fmt}

User question:
{q_clean}

Context (excerpts):
{context}
""".strip()

            draft = self.llm.ask(prompt).strip()
            if draft.startswith("⚠️"):
                return draft

            # Insert confidence after answer header if possible
            return f"{draft}\n\n{conf_block}".strip()

        except Exception as e:
            logging.exception(f"Error during answer generation: {e}")
            return "⚠️ Something went wrong while generating the answer. Please try again."
