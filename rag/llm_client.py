import time
import logging
from typing import Optional

from openai import OpenAI

# Best-effort imports for OpenAI python exceptions (names can vary by version)
try:
    from openai import (
        APIError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
    )
except Exception:  # pragma: no cover
    APIError = Exception
    RateLimitError = Exception
    APITimeoutError = Exception
    APIConnectionError = Exception
    BadRequestError = Exception
    AuthenticationError = Exception
    PermissionDeniedError = Exception
    NotFoundError = Exception

logging.basicConfig(level=logging.INFO)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_output_tokens: int = 650,
        temperature: float = 0.0,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def _extract_output_text(self, response) -> str:
        """
        Prefer `response.output_text` (OpenAI convenience field).
        Fallback to walking the response structure if needed.
        """
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # Fallback (defensive): try to find any text content
        try:
            outputs = getattr(response, "output", None) or []
            for o in outputs:
                content = getattr(o, "content", None) or []
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t.strip():
                        return t.strip()
        except Exception:
            pass

        return ""

    def ask(self, prompt: str, attempts: int = 5) -> str:
        """
        Calls OpenAI Responses API with retries on transient failures.
        """
        last_text: Optional[str] = None

        for attempt in range(1, attempts + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    # store=False  # optional; keep off unless you explicitly need it
                )

                text = self._extract_output_text(response)
                if text:
                    return text

                # If empty output, treat as transient and retry
                last_text = last_text or ""
                time.sleep(0.5 * attempt)

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                logging.warning(f"LLM transient error (attempt {attempt}/{attempts}): {type(e).__name__}")
                time.sleep(0.8 * attempt)

            except APIError as e:
                # APIError often includes 5xx; retry. If it's not retryable, it will likely be a BadRequestError etc.
                logging.warning(f"LLM API error (attempt {attempt}/{attempts}): {type(e).__name__}")
                time.sleep(0.8 * attempt)

            except (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError) as e:
                # These are usually NOT transient (bad key, bad model name, invalid request)
                logging.error(f"LLM non-retryable error: {type(e).__name__}")
                return (
                    f"⚠️ LLM request failed ({type(e).__name__}). "
                    "Check your OPENAI_API_KEY / model name and try again."
                )

            except Exception as e:
                # Unknown error: retry a bit, but log it
                logging.exception(f"LLM unexpected error (attempt {attempt}/{attempts}): {e}")
                time.sleep(0.8 * attempt)

        return last_text or (
            "⚠️ The LLM service is temporarily unavailable. "
            "Please try again shortly."
        )
