import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)


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

        # Detect environment (default = local)
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

    def _format_excerpts(self, retrieved: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved items into labeled sources the model can cite as (Title, Year).
        """
        parts: List[str] = []
        for i, item in enumerate(retrieved, start=1):
            title = (item.get("title") or "").strip()
            year = item.get("year", "")
            pmid = item.get("pmid", "")
            evidence = item.get("evidence_level", "")

            source_label = ""
            if title and year:
                source_label = f"{title} ({year})"
            elif title:
                source_label = title
            else:
                source_label = f"Source {i}"

            text = (item.get("text") or "").strip()

            parts.append(
                f"SOURCE {i}: {source_label}\n"
                f"PMID: {pmid} | Year: {year} | Evidence level: {evidence}\n"
                f"EXCERPT:\n{text}"
            )

        return "\n\n".join(parts).strip()

    def answer(self, question: str, chat_history: Optional[list] = None, k: int = 5) -> str:
        try:
            chat_context = self._format_chat_history(chat_history)

            retrieved = self.vector_store.search(question, k=k)

            # If vector store still returns strings, convert to dict format (no metadata)
            if retrieved and isinstance(retrieved[0], str):
                # preserve old-style error string behavior if it exists
                if len(retrieved) == 1 and retrieved[0].startswith("⚠️"):
                    return retrieved[0]
                retrieved = [{"text": t} for t in retrieved]

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

OUTPUT FORMAT (must follow exactly):

1) Summary (paraphrase):
- Provide a short, clear summary that paraphrases ONLY what is supported by the excerpts.
- After each key sentence, cite using (Title, Year). Example: (Smith et al., 2021)

2) Verbatim evidence (word-for-word):
- Provide 3–6 short direct quotes copied exactly from the excerpts.
- Each quote must be attributed with (Title, Year, PMID).
- Keep each quote short (1–2 sentences maximum).

Rules:
- Use ONLY the retrieved excerpts. Do not use outside knowledge.
- If insufficient information is present, respond exactly:
  Not enough evidence in the retrieved sources.
- Do not invent guideline names, statistics, or recommendations.

Answer:
""".strip()

            return self.llm.ask(prompt)

        except Exception as e:
            logging.error(f"Error during answer generation: {e}")
            return (
                "⚠️ Something went wrong while generating the answer. "
                "Please check your internet connection and try again."
            )
