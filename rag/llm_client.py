# rag/llm_client.py
import time
import logging
from typing import Any, Optional, List

from openai import OpenAI
from openai import BadRequestError, RateLimitError, APIConnectionError, InternalServerError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMClient:
    """
    OpenAI Responses API client with:
    - robust text extraction across response shapes
    - safe retry strategy
    - fallback when reasoning params cause empty output or incompatibility
    """

    def __init__(self, api_key: str, model: str, max_output_tokens: int = 1500):  # INCREASED from 700
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def _extract_text_from_any(self, obj: Any) -> str:
        """
        Extract assistant text from OpenAI Responses API response
        across different SDK / payload shapes.
        """
        # 1) best-case property from SDK
        try:
            t = getattr(obj, "output_text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            pass

        # 2) convert to dict if possible
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

        # 3) sometimes there's a top-level 'text'
        txt = data.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        if isinstance(txt, dict):
            v = txt.get("value") or txt.get("text")
            if isinstance(v, str) and v.strip():
                return v.strip()

        # ✅ handle list `text: [...]`
        if isinstance(txt, list):
            collected: List[str] = []
            for it in txt:
                if isinstance(it, str) and it.strip():
                    collected.append(it.strip())
                elif isinstance(it, dict):
                    v = it.get("text") or it.get("value")
                    if isinstance(v, str) and v.strip():
                        collected.append(v.strip())
            if collected:
                return "\n".join(collected).strip()

        # 4) standard Responses: output -> content blocks
        outputs = data.get("output", [])
        collected: List[str] = []

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
                # Common block shape:
                # { "type": "output_text", "text": "..." }
                if x.get("type") == "output_text":
                    t = x.get("text")
                    if isinstance(t, str) and t.strip():
                        collected.append(t.strip())

                # Another shape:
                # { "text": { "value": "..." } }
                t1 = x.get("text")
                if isinstance(t1, str) and t1.strip():
                    collected.append(t1.strip())
                elif isinstance(t1, dict):
                    v = t1.get("value") or t1.get("text")
                    if isinstance(v, str) and v.strip():
                        collected.append(v.strip())

                for key in ("content", "output", "message", "messages"):
                    if key in x:
                        walk(x[key])

        walk(outputs)

        if not collected:
            # try again in case response uses slightly different nesting
            walk(data.get("output"))
            walk(data.get("text"))

        return "\n".join([c for c in collected if c]).strip()

    def ask(self, prompt: str, attempt: int = 1, last_text: Optional[str] = None) -> str:
        """
        Calls OpenAI Responses API with retries and safe fallbacks.
        """
        if attempt > 5:
            return last_text or "⚠️ The LLM service is temporarily unavailable. Please try again shortly."

        def _call(with_reasoning: bool):
            kwargs = dict(
                model=self.model,
                input=prompt,
                max_output_tokens=self.max_output_tokens,
            )
            # Some models/accounts support this; if not, fallback below
            if with_reasoning:
                kwargs["reasoning"] = {"effort": "minimal"}
            return self.client.responses.create(**kwargs)

        try:
            # 1) Attempt with reasoning (fast on GPT-5 family when supported)
            resp = _call(with_reasoning=True)

            status = getattr(resp, "status", None)
            err = getattr(resp, "error", None)
            if err:
                logger.error(f"LLM response error field present: {err}")
            if status and status != "completed":
                logger.warning(f"LLM status not completed: {status}")

            text = self._extract_text_from_any(resp)
            if text:
                return text

            # 2) Fallback: retry once without reasoning
            resp2 = _call(with_reasoning=False)
            status2 = getattr(resp2, "status", None)
            err2 = getattr(resp2, "error", None)
            if err2:
                logger.error(f"LLM response error field present (no-reasoning): {err2}")
            if status2 and status2 != "completed":
                logger.warning(f"LLM status not completed (no-reasoning): {status2}")

            text2 = self._extract_text_from_any(resp2)
            if text2:
                return text2

            logger.warning("LLM returned no text (both attempts).")
            time.sleep(0.5 * attempt)
            return self.ask(prompt, attempt + 1, last_text=last_text)

        except BadRequestError as e:
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
```

---

## Key Changes:

### 1. **Aggressive Prompt** (lines 363-456 in qa_pipeline.py)
- Very explicit "DO NOT say 'not provided'" instructions
- Clear examples of what to extract
- Forces comprehensive extraction
- Multiple warnings against conservative behavior

### 2. **Context Preview Logging** (lines 323-333)
- Logs first/last 20 lines of context
- Helps debug what's actually being sent to LLM
- Can be commented out in production

### 3. **Increased Max Tokens** (llm_client.py line 17)
- From 700 → 1500 tokens
- Allows longer, more detailed responses

### 4. **Larger Context Window** (lines 17-20)
- 8 sources, 3 chunks each
- 1200 chars per chunk
- 8500 total chars

---

## Test Now:

Ask: **"What are the complications of radioactive iodine therapy?"**

You should get a response like:
```
AI Overview
Radioactive iodine therapy for thyroid cancer can cause several well-documented adverse effects. Salivary gland dysfunction occurs in 5-86% of patients in a dose-dependent manner. Other reported complications include xerostomia, taste and smell impairment, sialadenitis, and increased risk of secondary malignancies including therapy-related leukemia...

Known Complications/Adverse Effects:
- Salivary gland dysfunction: Occurs in 5-86% of patients, dose-dependent
- Xerostomia (dry mouth): Increases risk of dental caries
- Taste and smell impairment: Reported complication of RAI treatment
- Sialadenitis: Inflammation of salivary glands
- Secondary malignancy risk: Including therapy-related acute myeloid leukemia
- Reproductive effects: Studies show no deleterious effect on pregnancy outcomes
...
