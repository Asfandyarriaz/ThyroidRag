import time
from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def ask(self, prompt: str, attempt: int = 1, last_text: str | None = None) -> str:
        """
        Calls OpenAI Responses API with retries on transient failures.
        """
        if attempt > 5:
            return last_text or (
                "⚠️ The LLM service is temporarily unavailable. "
                "Please check your internet connection and try again shortly."
            )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )

            text = getattr(response, "output_text", None)
            if text:
                return text

            # If for some reason output_text is empty, retry
            time.sleep(0.5 * attempt)
            return self.ask(prompt, attempt + 1, last_text=last_text)

        except Exception:
            # Treat as transient and retry with backoff
            time.sleep(0.5 * attempt)
            return self.ask(prompt, attempt + 1, last_text=last_text)
