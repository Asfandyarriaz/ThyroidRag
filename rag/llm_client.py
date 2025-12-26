# rag/llm_client.py
import time
import logging
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


class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = (model or "").strip()  # IMPORTANT: strip whitespace

    def ask(self, prompt: str, attempt: int = 1, last_text: str | None = None) -> str:
        if not self.model:
            return "⚠️ OPENAI_MODEL is missing. Set it in Streamlit Secrets as OPENAI_MODEL."

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=800,  # supported in Responses API :contentReference[oaicite:1]{index=1}
            )
            text = getattr(resp, "output_text", None)
            return text.strip() if text else "⚠️ No text returned from the LLM."

        # ---- Non-retryable errors (don’t loop) ----
        except BadRequestError as e:
            logger.error("OpenAI 400 BadRequestError: %s", str(e))
            return (
                "⚠️ OpenAI rejected the request (400 Bad Request).\n\n"
                "Most common causes:\n"
                "- OPENAI_MODEL is wrong/blank (check Streamlit Secrets)\n"
                "- prompt/context too large\n\n"
                "Open the Streamlit logs for the exact error message."
            )
        except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
            logger.error("OpenAI auth/permission/model error: %s", str(e))
            return (
                "⚠️ OpenAI auth/model error.\n\n"
                "Check:\n"
                "- OPENAI_API_KEY is valid\n"
                "- OPENAI_MODEL exists and your key has access"
            )

        # ---- Retryable errors ----
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            logger.warning("OpenAI retryable error (attempt %s): %s", attempt, str(e))
            if attempt >= 5:
                return last_text or (
                    "⚠️ The LLM service is temporarily unavailable (rate limit/network/server). "
                    "Please try again shortly."
                )
            time.sleep(0.7 * attempt)
            return self.ask(prompt, attempt + 1, last_text=last_text)

        except Exception as e:
            logger.exception("Unexpected LLM error: %s", str(e))
            return "⚠️ Unexpected error calling the LLM. Check Streamlit logs."
