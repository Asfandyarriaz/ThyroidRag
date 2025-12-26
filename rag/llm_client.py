import time
import logging
from typing import Optional

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

    def _extract_text(self, resp) -> str:
        """
        Robustly extract text from OpenAI Responses API response.
        - Prefer resp.output_text
        - Fallback: walk resp.output[].content[].text
        """
        # 1) Convenience field
        t = getattr(resp, "output_text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()

        # 2) Structured output
        out = getattr(resp, "output", None)
        if isinstance(out, list):
            texts = []
            for item in out:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for c in content:
                    # Some SDKs store text in c.text, others in c.get("text") if dict
                    if hasattr(c, "text"):
                        ct = getattr(c, "text", None)
                        if isinstance(ct, str) and ct.strip():
                            texts.append(ct.strip())
                    elif isinstance(c, dict):
                        ct = c.get("text")
                        if isinstance(ct, str) and ct.strip():
                            texts.append(ct.strip())
            if texts:
                return "\n".join(texts).strip()

        return ""

    def ask(self, prompt: str, attempts: int = 3) -> str:
        if not self.model:
            return "⚠️ OPENAI_MODEL is missing. Set it in Streamlit Secrets as OPENAI_MODEL."

        last_err: Optional[str] = None

        for attempt in range(1, attempts + 1):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=0.0,
                    max_output_tokens=800,
                )

                text = self._extract_text(resp)
                if text:
                    return text

                # Request succeeded but no text; treat as transient once or twice
                last_err = "No text returned from the LLM."
                time.sleep(0.4 * attempt)

            except BadRequestError as e:
                logger.error("OpenAI 400 BadRequestError: %s", str(e))
                return (
                    "⚠️ OpenAI rejected the request (400 Bad Request).\n\n"
                    "Common causes:\n"
                    "- OPENAI_MODEL is wrong/blank\n"
                    "- prompt/context too large\n\n"
                    "Check Streamlit logs for details."
                )

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
