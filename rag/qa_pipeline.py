import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)


class QAPipeline:
    def __init__(self, embedder, vector_store, llm_client,
                 rag_instruction_file="instructions/rag_instructions.txt",
                 agent_instruction_file="instructions/agent_instructions.txt"):
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

    def _load_instruction(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            logging.warning(f"Instruction file {file_path} not found.")
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"Failed to load instruction file {file_path}: {e}")
            return ""

    def _combine_instructions(self) -> str:
        instructions = ""
        if self.agent_instructions:
            instructions += f"{self.agent_instructions}\n\n"
        if self.rag_instructions:
            instructions += f"{self.rag_instructions}"
        return instructions.strip()

    def _format_chat_history(self, chat_history: Optional[list]) -> str:
        if not chat_history:
            return ""
        lines = []
        for msg in chat_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    def _format_excerpts(self, retrieved):
    parts = []
    for i, item in enumerate(retrieved, start=1):
        title = (item.get("title") or "").strip()
        year = item.get("year")
        pmid = item.get("pmid")
        evidence = item.get("evidence_level")

        # Short label the model can use consistently
        source_label = f"{title} ({year})" if title and year else (title or f"Source {i}")

        text = (item.get("text") or "").strip()

        parts.append(
            f"SOURCE {i}: {source_label}\n"
            f"PMID: {pmid} | Year: {year} | Evidence level: {evidence}\n"
            f"EXCERPT:\n{text}"
        )
    return "\n\n".join(parts).strip()

    def answer(self, question: str, chat_history: list = None, k: int = 5) -> str:
        try:
            chat_context = self._format_chat_history(chat_history)

            retrieved = self.vector_store.search(question, k=k)

            # If your vector_store still returns strings, this will adapt:
            # - strings become {"text": "..."}
            if retrieved and isinstance(retrieved[0], str):
                # Preserve your old error handling
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
- If the excerpts do not contain enough information, respond exactly:
  Not enough evidence in the retrieved sources.
- Do not invent guideline names, statistics, or recommendations.
- Do not cite sources not present in the excerpts.

Answer:
""".strip()


            response = self.llm.ask(prompt)
            return response

        except Exception as e:
            logging.error(f"Error during answer generation: {e}")
            return "⚠️ Something went wrong while generating the answer. Please check your internet connection and try again."
