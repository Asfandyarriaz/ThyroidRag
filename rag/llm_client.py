import time
import logging
from typing import Optional, Any, List

from openai import OpenAI
from openai import (
    BadRequestError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = (model or "").strip()

    def _as_dict(self, resp: Any) -> dict:
        """
        Convert OpenAI SDK response object to a plain dict safely.
        """
        if isinstance(resp, dict):
            return resp
        if hasattr(resp, "model_dump"):
            try:
                return resp.model_dump()
            except Exception:
                pass
        if hasattr(resp, "to_dict"):
            try:
                return resp.to_dict()
            except Exception:
                pass
        # last resort
        try:
            return dict(resp)  # may fail
        except Exception:
            return {}

    def _collect_texts(self, obj: Any, out: List[str]) -> None:
        """
        Recursively collect text fields from nested dict/list structures.
        This reliably catches Responses API formats like:
          output -> [ { content: [ { type: "output_text", text: "..." } ] } ]
        """
        if obj is None:
            return

        if isinstance(obj, dict):
            # Most common: {"type":"output_text","text":"..."}
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                # If it has a type, prefer output_text/text-like blocks
                typ = (obj.get("type") or "").lower()
                if typ in ("output_text", "text", "message", ""):
                    out.append(t.strip())

            # Recurse
            for v in obj.values():
                self._collect_texts(v, out)

        elif isinstance(obj, list):
            for it in obj:
                self._collect_texts(it, out)

    def _extract_text(self, resp: Any) -> str:
        # 1) Prefer the convenience field when present
        t = getattr(resp, "output_text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()

        # 2) Try dict form
        d = self._as_dict(resp)
        t2 = d.get("output_text")
        if isinstance(t2, str) and t2.strip():
            return t2.strip()

        # 3) Deep-collect from output/content blocks
        texts: List[str] = []
        self._collect_texts(d.get("output"), texts)

        # remove duplicates while preserving order
        seen = set()
        uniq = []
        for s in texts:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        return "\n".join(uniq).strip()

    def ask(self, prompt: str, attempts: int = 3) -> str:
        if not self.model:
            return "⚠️ OPENAI_MODEL is missing. Set it in Streamlit Secrets as OPENAI_MODEL."

        last_err: Optional[str] = None

        for attempt in range(1, attempts + 1):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=900,
                )

                text = self._extract_text(resp)
                if text:
                    return text

                last_err = "No text returned from the LLM."
                # Helpful log (no secrets): log top-level keys so we can see shape
                try:
                    d = self._as_dict(resp)
                    logger.warning("LLM returned no text. Top-level keys: %s", list(d.keys()))
                except Exception:
                    pass

                time.sleep(0.4 * attempt)

            except BadRequestError as e:
                logger.error("OpenAI 400 BadRequestError: %s", str(e))
                return "⚠️ OpenAI rejected the request (400). Check Streamlit logs for the exact reason."

            except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
                logger.error("OpenAI auth/model error: %s", str(e))
                return (
                    "⚠️ OpenAI auth/model error.\n\n"
                    "Check:\n"
                    "- OPENAI_API_KEY is valid\n"
                    "- OPENAI_MODEL exists and your key has access"
                )

            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                logger.warning("OpenAI retryable error (attempt %s/%s): %s", attempt, attempts, str(e))
                last_err = "LLM temporarily unavailable."
                time.sleep(0.8 * attempt)

            except Exception as e:
                logger.exception("Unexpected LLM error: %s", str(e))
                last_err = "Unexpected LLM error."
                time.sleep(0.8 * attempt)

        return f"⚠️ {last_err or 'The LLM service is temporarily unavailable. Please try again shortly.'}"
