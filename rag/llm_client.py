# rag/llm_client.py
import time
import logging
from typing import Any, Optional

from openai import OpenAI
from openai import BadRequestError, RateLimitError, APIConnectionError, InternalServerError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMClient:
    def __init__(self, api_key: str, model: str, max_output_tokens: int = 700):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def _extract_text_from_any(self, obj: Any) -> str:
        """
        Tries hard to extract model text from different Responses shapes.
        Works with both SDK objects and dict-like responses.
        """
        # 1) SDK convenience property (best case)
        try:
            t = getattr(obj, "output_text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            pass

        # 2) Convert SDK object -> dict if possible
        data = None
        if isinstance(obj, dict):
            data = obj
        else:
            try:
                if hasattr(obj, "model_dump"):
                    data = obj.model_dump()
                elif hasattr(obj, "dict"):
                    data = obj.dict()
            except Exception:
                data = None

        if not isinstance(data, dict):
            return ""

        # 3) Sometimes there's a top-level `text` container
        txt = data.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        if isinstance(txt, dict):
            for k in ("value", "text"):
                v = txt.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        # 4) Standard Responses structure: output -> message -> content -> output_text
        outputs = data.get("output", [])
        collected = []

        def walk(x: Any):
            if x is None:
                return
            if isinstance(x, str):
                s = x.strip()
                if s:
                    collected.append(s)
                return
            if isinstance(x, list):
                for it in x:
                    walk(it)
                return
            if isinstance(x, dict):
                # common patterns:
                # {type: "output_text", text: "..."} OR {text: {value: "..."}}
                t1 = x.get("text")
                if isinstance(t1, str) and t1.strip():
                    collected.append(t1.strip())
                elif isinstance(t1, dict):
                    v = t1.get("value") or t1.get("text")
                    if isinstance(v, str) and v.strip():
                        collected.append(v.strip())

                # recurse into likely containers
                for key in ("content", "output", "message", "messages"):
                    if key in x:
                        walk(x[key])

        walk(outputs)

        # As a last resort, walk the top-level `output` and `text` only
        if not collected:
            walk(data.get("output"))
            walk(data.get("text"))

        # Deduplicate lightly
        out = "\n".join([c for c in collected if c])
        return out.strip()

    def ask(self, prompt: str, attempt: int = 1, last_text: Optional[str] = None) -> str:
        if attempt > 5:
            return last_text or "⚠️ The LLM service is temporarily unavailable. Please try again shortly."

        try:
            # IMPORTANT: do NOT pass temperature for gpt-5-nano (your logs show 400 when temperature is sent)
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=self.max_output_tokens,
                # Optional speed-up for GPT-5 family (safe if supported; if not supported it may 400)
                reasoning={"effort": "minimal"},
            )

            text = self._extract_text_from_any(resp)
            if text:
                return text

            # If no text, retry once without reasoning param (some accounts/models vary)
            if attempt == 1:
                resp2 = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=self.max_output_tokens,
                )
                text2 = self._extract_text_from_any(resp2)
                if text2:
                    return text2

            logger.warning("LLM returned no text.")
            time.sleep(0.5 * attempt)
            return self.ask(prompt, attempt + 1, last_text=last_text)

        except BadRequestError as e:
            # Don't retry 400s (usually model/params/prompt)
            msg = str(e)
            logger.error(f"OpenAI 400 BadRequestError: {msg}")
            return (
                "⚠️ OpenAI rejected the request (400 Bad Request).\n\n"
                "Common causes:\n"
                "- OPENAI_MODEL is wrong/blank\n"
                "- prompt/context too large\n"
                "- unsupported params for the selected model\n\n"
                "Check Streamlit logs for details."
            )

        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            logger.warning(f"Transient LLM error (attempt {attempt}/5): {e}")
            time.sleep(0.5 * attempt)
            return self.ask(prompt, attempt + 1, last_text=last_text)

        except Exception as e:
            logger.warning(f"Unknown LLM error (attempt {attempt}/5): {e}")
            time.sleep(0.5 * attempt)
            return self.ask(prompt, attempt + 1, last_text=last_text)
