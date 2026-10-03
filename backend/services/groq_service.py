import logging
from typing import Optional
from groq import Groq, GroqError

logger = logging.getLogger(__name__)

class GroqService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)

    def verify_scan(self, image_b64: str, prompt: str, model: str = "qwen/qwen3.8-27b") -> Optional[str]:
        if not self.client:
            logger.error("Groq client not initialized (missing API key).")
            return None

        try:
            if not image_b64.startswith("data:"):
                image_url = f"data:image/jpeg;base64,{image_b64}"
            else:
                image_url = image_b64

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=30.0
            )
            return response.choices[0].message.content
        except GroqError as e:
            logger.error("Groq verification failed due to GroqError.")
            return None
        except Exception as e:
            logger.error("Unexpected error in Groq verification.")
            return None

    def generate_report(self, prompt: str) -> Optional[dict]:
        """Generate structured clinical report via Groq Llama / Qwen."""
        if not self.client:
            logger.error("Groq client not initialized (missing API key).")
            return None
        import json
        models_to_try = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        for model in models_to_try:
            try:
                max_toks = 700 if "qwen" in model else 1500
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a clinical diagnostic reporting AI. Output strictly valid JSON matching the requested schema without markdown code fences."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=max_toks,
                    timeout=35.0
                )
                raw = response.choices[0].message.content
                if raw:
                    cleaned = raw.strip()
                    if "```" in cleaned:
                        parts = cleaned.split("```")
                        for p in parts:
                            p = p.strip()
                            if p.startswith("json"):
                                p = p[4:].strip()
                            try:
                                data = json.loads(p)
                                logger.info("Report generated successfully using Groq model %s", model)
                                return data
                            except Exception:
                                continue
                    data = json.loads(cleaned)
                    logger.info("Report generated successfully using Groq model %s", model)
                    return data
            except Exception as e:
                logger.warning("Groq report generation with %s failed: %s", model, e)
                continue
        return None
