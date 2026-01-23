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

# Kept ONLY as heuristic fallback if LLM scope judge fails (normal Q&A path)
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

LEVEL_SYNONYMS: Dict[int, List[str]] = {
    1: ["guideline", "guidelines", "practice guideline", "clinical guideline", "consensus", "consensus conference"],
    2: ["systematic review", "systematic reviews", "meta analysis", "meta-analysis", "network meta", "network meta analysis"],
    3: ["rct", "rcts", "randomized", "randomised", "randomized controlled trial", "controlled clinical trial", "phase iii", "phase iv"],
    4: ["clinical trial", "clinical trials", "trial protocol", "equivalence trial", "phase i", "phase ii", "non randomized trial", "non-randomized trial"],
    5: ["cohort", "cohort study", "cohort studies", "prospective", "retrospective"],
    6: ["case control", "case-control", "case control study", "case-control study"],
    7: ["case report", "case reports", "case series", "series", "personal narrative"],
}

HIERARCHY_HIGH_WORDS = ["higher", "above", "better", "stronger", "top", "best", "greater"]
HIERARCHY_LOW_WORDS = ["lower", "below", "worse", "weaker", "less"]

MAX_SOURCES = 6
MAX_CHUNKS_PER_SOURCE = 2
MAX_EXCERPT_CHARS = 900
MAX_TOTAL_CONTEXT_CHARS = 6500


def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\s]+", " ", t)
    t = re.sub(r"[^\w\s\-\[\]]", "", t)  # keep - and [] for patterns like [credibility check]
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

        # caches
        self._intent_cache: Dict[str, Tuple[str, float, str]] = {}
        self._scope_cache: Dict[str, Tuple[bool, float, str]] = {}

    # -------------------- instruction loading --------------------

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

    # -------------------- routing helpers --------------------

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

    # -------------------- JSON extraction helper --------------------

    def _extract_json_object(self, s: str) -> Optional[str]:
        if not s:
            return None
        s = s.strip()
        if s.startswith("{") and s.endswith("}"):
            return s
        m = re.search(r"\{.*\}", s, flags=re.S)
        return m.group(0) if m else None

    # -------------------- LLM intent router (assistant vs medical) --------------------

    def _llm_intent_router(self, q: str) -> Tuple[Optional[str], float, str]:
        q_clean = (q or "").strip()
        if not q_clean:
            return None, 0.0, "Empty question."

        qn = _norm(q_clean)
        if qn in self._intent_cache:
            label, conf, rat = self._intent_cache[qn]
            return label, conf, rat

        prompt = f"""
You are an intent router for an app called "Thyroid Cancer RAG Assistant".

Pick EXACTLY ONE label:
- "assistant": the user asks about the chatbot/app itself (who are you, what can you do, how it works, help using it)
- "medical": the user asks about thyroid cancer topics
- "other": neither

Important:
- If user mentions cancer (e.g., "what is this cancer"), choose "medical" not "assistant".
- Choose "assistant" only when clearly about the chatbot/app.

Return ONLY JSON:
{{
  "label": "assistant" | "medical" | "other",
  "confidence": 0 to 1,
  "rationale": "<= 20 words"
}}

User message:
{q_clean}
""".strip()

        try:
            raw = (self.llm.ask(prompt) or "").strip()
            js = self._extract_json_object(raw)
            if not js:
                return None, 0.0, "LLM did not return JSON."
            data = json.loads(js)

            label = data.get("label")
            conf = data.get("confidence", 0.0)
            rat = data.get("rationale", "") or ""

            if label not in ("assistant", "medical", "other"):
                return None, 0.0, "Invalid label."

            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            conf_f = max(0.0, min(1.0, conf_f))
            rat = str(rat)[:200]

            self._intent_cache[qn] = (label, conf_f, rat)
            return label, conf_f, rat
        except Exception as e:
            logging.warning(f"LLM intent router failed: {e}")
            return None, 0.0, "LLM intent router error."

    def _is_assistant_question(self, q: str) -> bool:
        label, _conf, _rat = self._llm_intent_router(q)
        if label is None:
            qn = _norm(q)
            return any(x in qn for x in ["who are you", "what can you do", "how does this work", "help"])
        return label == "assistant"

    # -------------------- scope check (LLM-based, normal Q&A only) --------------------

    def _llm_scope_judge(self, q: str) -> Tuple[Optional[bool], float, str]:
        q_clean = (q or "").strip()
        if not q_clean:
            return None, 0.0, "Empty question."

        qn = _norm(q_clean)
        if qn in self._scope_cache:
            ok, conf, rat = self._scope_cache[qn]
            return ok, conf, rat

        prompt = f"""
You are a scope classifier for "Thyroid Cancer RAG Assistant".

Goal:
Decide whether the user's question is IN SCOPE for thyroid cancer.

Return ONLY JSON:
{{
  "in_scope": true/false,
  "confidence": 0 to 1,
  "rationale": "<= 20 words"
}}

User question:
{q_clean}
""".strip()

        try:
            raw = (self.llm.ask(prompt) or "").strip()
            js = self._extract_json_object(raw)
            if not js:
                return None, 0.0, "LLM did not return JSON."
            data = json.loads(js)

            in_scope = data.get("in_scope", None)
            conf = data.get("confidence", 0.0)
            rat = data.get("rationale", "") or ""

            if not isinstance(in_scope, bool):
                return None, 0.0, "LLM JSON missing boolean in_scope."

            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            conf_f = max(0.0, min(1.0, conf_f))
            rat = str(rat)[:200]

            self._scope_cache[qn] = (in_scope, conf_f, rat)
            return in_scope, conf_f, rat
        except Exception as e:
            logging.warning(f"LLM scope judge failed: {e}")
            return None, 0.0, "LLM scope judge error."

    def _heuristic_in_scope(self, q: str) -> bool:
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

    def _is_in_scope(self, q: str) -> bool:
        in_scope, _conf, _rat = self._llm_scope_judge(q)
        if in_scope is None:
            return self._heuristic_in_scope(q)
        return bool(in_scope)

    # -------------------- credibility mode --------------------

    def _is_credibility_check(self, q: str) -> bool:
        q_strip = (q or "").strip().lower()
        return (
            q_strip.startswith("credibility_check:")
            or q_strip.startswith("check credibility:")
            or q_strip.startswith("[credibility check]")
        )

    def _extract_claim(self, q: str) -> str:
        s = (q or "").strip()
        lower = s.lower()
        if lower.startswith("credibility_check:"):
            return s[len("credibility_check:"):].strip()
        if lower.startswith("check credibility:"):
            return s[len("check credibility:"):].strip()
        if lower.startswith("[credibility check]"):
            return s[len("[credibility check]"):].strip()
        return s

    # -------------------- level filter parsing --------------------

    def _parse_level_filter(self, q: str) -> Optional[List[int]]:
        qn = _norm(q)
        if not qn:
            return None

        levels_found = set()

        for lvl, kws in LEVEL_SYNONYMS.items():
            for kw in kws:
                if kw in qn:
                    levels_found.add(lvl)

        m_range = re.search(r"\blevels?\s*([1-7])\s*-\s*([1-7])\b", qn)
        if m_range:
            a, b = int(m_range.group(1)), int(m_range.group(2))
            lo, hi = min(a, b), max(a, b)
            return list(range(lo, hi + 1))

        for n in re.findall(r"\blevel\s*([1-7])\b", qn):
            levels_found.add(int(n))

        m_comp = re.search(r"(?:(<=|>=|<|>)\s*)level\s*([1-7])\b", qn)
        if m_comp:
            op, n = m_comp.group(1), int(m_comp.group(2))
            if op == "<=":
                return list(range(1, n + 1))
            if op == "<":
                return list(range(1, n))
            if op == ">=":
                return list(range(n, 8))
            if op == ">":
                return list(range(n + 1, 8))

        m_nl = re.search(
            r"\b(higher|above|better|stronger|greater|lower|below|worse|weaker|less)\s+(?:than\s+)?level\s*([1-7])\b",
            qn
        )
        if not m_nl:
            m_nl = re.search(
                r"\b(higher|above|better|stronger|greater|lower|below|worse|weaker|less)\s+(?:than\s+)?([1-7])\b",
            )
        if m_nl:
            word = m_nl.group(1)
            n = int(m_nl.group(2))
            if word in HIERARCHY_HIGH_WORDS:
                return list(range(1, n + 1))
            if word in HIERARCHY_LOW_WORDS:
                return list(range(n, 8))

        return sorted(levels_found) if levels_found else None

    def _strip_level_filter_text(self, q: str) -> str:
        s = q or ""
        s = re.sub(r"\bfrom\s+(guidelines?|consensus)\s+only\b", "", s, flags=re.I)
        s = re.sub(r"\bguidelines?\s+only\b", "", s, flags=re.I)
        s = re.sub(
            r"\b(systematic review|meta[\s\-]?analysis|rcts?|randomi[sz]ed|clinical trials?|cohort|case[\s\-]?control|case reports?|case series)\s+only\b",
            "",
            s,
            flags=re.I,
        )
        s = re.sub(r"\bonly\s+from\s+levels?\s*[1-7](\s*-\s*[1-7])?\b", "", s, flags=re.I)
        s = re.sub(r"\bfrom\s+levels?\s*[1-7](\s*-\s*[1-7])?\b", "", s, flags=re.I)
        s = re.sub(r"\blevels?\s*[1-7](\s*-\s*[1-7])?\s*only\b", "", s, flags=re.I)
        s = re.sub(r"\blevels?\s*[1-7](\s*(and|,)\s*[1-7])+\b", "", s, flags=re.I)
        s = re.sub(r"\blevel\s*[1-7]\b", "", s, flags=re.I)
        s = re.sub(r"(<=|>=|<|>)\s*level\s*[1-7]\b", "", s, flags=re.I)
        s = re.sub(
            r"\b(higher|above|better|stronger|greater|lower|below|worse|weaker|less)\s+(than\s+)?level\s*[1-7]\b",
            "",
            s,
            flags=re.I
        )
        s = re.sub(
            r"\b(higher|above|better|stronger|greater|lower|below|worse|weaker|less)\s+(than\s+)?[1-7]\b",
            "",
            s,
            flags=re.I
        )
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # -------------------- confidence rating --------------------

    def _safe_int(self, x: Any) -> Optional[int]:
        try:
            if x is None or isinstance(x, bool):
                return None
            return int(float(x))
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

        parts = []
        for lvl in sorted(counts.keys()):
            parts.append(f"Level {lvl} ({EVIDENCE_LEVEL_INFO[lvl][0]}): {counts[lvl]}")
        breakdown = "; ".join(parts)

        return {"label": label, "score": score, "breakdown": breakdown}

    def _format_confidence_block(self, conf: Dict[str, Any]) -> str:
        return (
            "Confidence rating\n"
            f"- **{conf['label']}** ({conf['score']}/100)\n"
            f"- Based on retrieved evidence levels: {conf['breakdown']}"
        )

    # -------------------- paper mode --------------------

    def _extract_pmid(self, q: str) -> Optional[int]:
        m = re.search(r"\bpmid\b[:\s]*([0-9]{{6,10}})\b", (q or "").lower())
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

    # -------------------- context builder --------------------

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

    # -------------------- prompts --------------------

    def _prompt_normal(self, question: str, context: str, mode: str) -> str:
        if mode == "short":
            output_format = """A) Definition
- 1–2 bullets, each ending with (Title, Year).

B) Key sources
- 1–3 bullet items: (Title, Year)

(Do not include a Summary section in short mode.)"""
        elif mode == "evidence":
            output_format = """A) Definition
- 1–2 bullets, each ending with (Title, Year).

B) Summary
- 3–5 bullets, each ending with (Title, Year).

C) Verbatim evidence
- 3–5 direct quotes (<= 35 words each) as: "<quote>" (Title, Year, PMID: <PMID>)"""
        else:
            output_format = """A) Definition
- 1–2 bullets, each ending with (Title, Year).

B) Summary
- 3–5 bullets, each ending with (Title, Year)."""

        return f"""
{self.instruction_text}

You MUST answer using ONLY the excerpts in the context.

OUTPUT FORMAT (follow exactly):
{output_format}

User question:
{question}

Context (excerpts):
{context}
""".strip()

    # -------------------- STRICT credibility prompt + repair pass --------------------

    def _prompt_credibility(self, claim: str, context: str) -> str:
        return f"""
{self.instruction_text}

Task:
You are performing a credibility check for a third-party claim using ONLY the retrieved excerpts.

Hard rules (must follow):
- Use ONLY the excerpts below. No outside medical knowledge.
- Break the claim into atomic statements.
- For each statement, classify as:
  Supported / Contradicted / Not found in sources
- If you mark a statement Supported or Contradicted:
  - You MUST include at least one direct quote (<= 35 words) from the excerpts
  - The quote MUST be listed in the Evidence section
  - And you MUST cite it inline using (Title, Year) in the justification bullet
- If you mark a statement Not found in sources:
  - Do NOT add any extra information or inferences.
  - ONLY say that the sources do not mention it / do not address it.

Overall assessment rules:
- If ALL statements are Not found in sources -> Overall assessment MUST be: Not enough evidence in the retrieved sources
- If any statement is Contradicted and none Supported -> Overall assessment: Not supported
- If mix of Supported/Not found/Contradicted -> Overall assessment: Partially supported
- If all statements Supported -> Overall assessment: Supported

OUTPUT FORMAT (follow exactly):
Credibility check
- Overall assessment: <Supported / Partially supported / Not supported / Not enough evidence in the retrieved sources>
- Supported (from sources):
  - <statement>: <why> (Title, Year)
- Contradicted (by sources):
  - <statement>: <why> (Title, Year)
- Not found in sources:
  - <statement>

Evidence
- Up to 4 short quotes (<= 35 words), each as:
  "<quote>" (Title, Year, PMID: <PMID>)

Claim to verify:
{claim}

Context (excerpts):
{context}
""".strip()

    def _needs_credibility_repair(self, text: str) -> bool:
        if not text:
            return True

        t = text.lower()

        # Detect noncompliance: supported/contradicted bullets without quotes/evidence
        has_supported_bullets = bool(re.search(r"supported \(from sources\):\s*\n\s*-\s*(?!none)", text, flags=re.I))
        has_contra_bullets = bool(re.search(r"contradicted \(by sources\):\s*\n\s*-\s*(?!none)", text, flags=re.I))
        has_evidence_section = "evidence" in t
        has_quote = bool(re.search(r"\".{5,}\"", text))  # at least one "..."

        if (has_supported_bullets or has_contra_bullets) and (not has_evidence_section or not has_quote):
            return True

        # Not found section should not contain factual verbs suggesting inference
        m_nf = re.search(r"not found in sources:\s*(.*?)(\n\s*evidence|\Z)", text, flags=re.I | re.S)
        if m_nf:
            nf_block = m_nf.group(1).lower()
            forbidden = ["indicate", "suggest", "shows", "showed", "demonstrate", "demonstrates", "supports", "contradicts"]
            if any(w in nf_block for w in forbidden):
                return True

        # If it says "not enough evidence" but still puts extra facts elsewhere, also likely bad
        if "overall assessment" in t and "not enough evidence" in t:
            # allow, but still ensure it doesn't claim contradictions without evidence
            pass

        return False

    def _prompt_credibility_repair(self, bad_answer: str, claim: str, context: str) -> str:
        return f"""
You must correct the credibility-check output to comply with the rules.

Fixes required:
- Do not add any extra factual claims under "Not found in sources".
- Any Supported or Contradicted statement MUST be backed by at least one direct quote in the Evidence section.

Rewrite the answer from scratch using ONLY the excerpts and the required format.

Claim:
{claim}

Context (excerpts):
{context}

Bad answer to fix:
{bad_answer}
""".strip()

    # -------------------- answer --------------------

    def answer(self, question: str, chat_history: Optional[list] = None, k: int = 25) -> str:
        """
        NOTE: default k is now 25.
        """
        try:
            # LLM-based meta routing (assistant vs medical)
            if self._is_assistant_question(question):
                return (
                    "I’m the assistant inside **Thyroid Cancer RAG Assistant**.\n\n"
                    "- I answer **thyroid cancer** questions by retrieving relevant excerpts from your indexed dataset (Qdrant).\n"
                    "- Then I generate an evidence-grounded response using only those excerpts.\n\n"
                    "You can ask about: thyroid cancer types (PTC/FTC/MTC/ATC), nodules, ultrasound/TIRADS, biopsy/FNA, "
                    "staging, surgery, radioiodine (RAI), follow-up, recurrence, and prognosis."
                )

            requested_levels = self._parse_level_filter(question)
            q_clean = self._strip_level_filter_text(question)

            # -------------------- Credibility check (accept ANY claim) --------------------
            if self._is_credibility_check(q_clean):
                claim = self._extract_claim(q_clean)

                # retrieval with thyroid bias + fallback
                k_use = max(k, 12)
                retrieval_query = f"{claim} thyroid cancer"
                retrieved = self.vector_store.search(retrieval_query, k=k_use, levels=requested_levels)
                if not retrieved:
                    retrieved = self.vector_store.search(claim, k=k_use, levels=requested_levels)

                if not retrieved:
                    return "Not enough evidence in the retrieved sources."

                conf = self._compute_confidence(retrieved)
                conf_block = self._format_confidence_block(conf)

                context = self._build_context(retrieved)
                prompt = self._prompt_credibility(claim, context)
                draft = (self.llm.ask(prompt) or "").strip()
                if draft.startswith("⚠️"):
                    return draft

                # Repair pass if output violates your strict credibility rules
                if self._needs_credibility_repair(draft):
                    repair_prompt = self._prompt_credibility_repair(draft, claim, context)
                    draft2 = (self.llm.ask(repair_prompt) or "").strip()
                    if draft2 and not self._needs_credibility_repair(draft2):
                        draft = draft2
                    else:
                        # last resort: safe compliant output
                        draft = (
                            "Credibility check\n"
                            "- Overall assessment: Not enough evidence in the retrieved sources\n"
                            "- Supported (from sources):\n"
                            "  - None\n"
                            "- Contradicted (by sources):\n"
                            "  - None\n"
                            "- Not found in sources:\n"
                            f"  - {claim}\n\n"
                            "Evidence\n"
                            "- Not provided, as no excerpt supports or contradicts the claim."
                        )

                # Insert confidence block at top
                if "Credibility check" in draft:
                    draft = draft.replace("Credibility check", f"Credibility check\n\n{conf_block}", 1)
                else:
                    draft = f"Credibility check\n\n{conf_block}\n\n{draft}"

                return draft.strip()

            # -------------------- Normal Q&A scope gate --------------------
            if not self._is_in_scope(q_clean):
                return (
                    "I’m scoped to **thyroid cancer** questions only.\n\n"
                    "Try asking about thyroid nodules, ultrasound features, biopsy/FNA, thyroid cancer subtypes, staging, "
                    "surgery, radioiodine (RAI), follow-up, or recurrence."
                )

            # Paper lookup
            if self._is_paper_request(question):
                requested_pmid = self._extract_pmid(question)
                retrieved = self.vector_store.search(q_clean, k=max(k, 12), levels=requested_levels)
                return self._paper_lookup_response(retrieved, requested_pmid)

            mode = self._select_mode(q_clean)

            retrieval_query = q_clean
            if self._is_definition_question(q_clean):
                term = self._extract_term(q_clean)
                if term:
                    retrieval_query = f"definition of {term}"

            k_use = max(k, 8) if mode == "short" else max(k, 12)
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
            prompt = self._prompt_normal(q_clean, context, mode=mode)
            draft = (self.llm.ask(prompt) or "").strip()
            if draft.startswith("⚠️"):
                return draft

            if "B) Summary" in draft:
                draft = draft.replace("B) Summary", f"{conf_block}\n\nB) Summary", 1)
            elif "B) Key sources" in draft:
                draft = draft.replace("B) Key sources", f"{conf_block}\n\nB) Key sources", 1)
            else:
                draft = f"{draft}\n\n{conf_block}"

            return draft.strip()

        except Exception as e:
            logging.exception(f"Error during answer generation: {e}")
            return "⚠️ Something went wrong while generating the answer. Please try again."
