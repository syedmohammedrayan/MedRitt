import logging
import json
from typing import Optional, List
from pydantic import BaseModel, ValidationError
from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

class ReportQAResult(BaseModel):
    passes: bool
    issues: List[str]
    unsupported_claims: List[str]
    grounding_warnings: List[str]
    recommendation: str

class NVIDIAService:
    def __init__(self, api_key: Optional[str] = None, model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"):
        self.api_key = api_key
        self.model = model
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://integrate.api.nvidia.com/v1"
            )

    def verify_report(self, diagnostic_result: dict, generated_report: str) -> Optional[ReportQAResult]:
        if not self.client:
            logger.error("NVIDIA service not initialized (missing API key).")
            return None

        prompt = f'''
You are a secondary medical report QA assistant.
Your job is to verify the generated report against the structured diagnostic result.
Do NOT invent new diagnoses or change the diagnostic result.

Diagnostic Result:
{json.dumps(diagnostic_result, indent=2)}

Generated Report:
{generated_report}

Analyze the report for unsupported claims, grounding issues, or fabricated patient history.
Output your response as JSON matching exactly this schema:
{{
  "passes": true,
  "issues": ["list of issues"],
  "unsupported_claims": ["list of unsupported claims"],
  "grounding_warnings": ["list of warnings"],
  "recommendation": "accept or flag"
}}
'''
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical QA assistant. Output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2048,
                timeout=60.0
            )
            content = response.choices[0].message.content
            if not content:
                logger.error("NVIDIA returned empty QA response.")
                return None
            logger.info("NVIDIA Raw Content: " + repr(content))

            import re
            cleaned = content.strip()
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                cleaned = match.group(0)

            parsed = json.loads(cleaned)
            result = ReportQAResult(**parsed)
            return result
        except OpenAIError as e:
            logger.error("NVIDIA QA verification failed due to OpenAIError: " + str(e))
            return None
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("NVIDIA QA verification returned malformed JSON: " + str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error in NVIDIA QA verification: " + str(e))
            return None

    def generate_report(self, prompt: str, image_bytes: Optional[bytes] = None) -> Optional[dict]:
        """Generate structured clinical report via NVIDIA NIM (vision or text)."""
        if not self.client:
            logger.error("NVIDIA client not initialized (missing API key).")
            return None
        try:
            image_b64 = None
            if image_bytes:
                import base64
                image_b64 = base64.b64encode(image_bytes).decode("ascii")

            models_to_try = []
            if image_b64:
                models_to_try.append(("meta/llama-3.2-11b-vision-instruct", True))
            models_to_try.extend([
                (self.model, False),
                ("meta/llama-3.2-11b-vision-instruct", False),
                ("mistralai/mistral-large-2-instruct", False),
            ])

            for m, use_vision in models_to_try:
                try:
                    if use_vision and image_b64:
                        user_content = [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    else:
                        user_content = prompt

                    response = self.client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": "You are a clinical diagnostic reporting AI. Output strictly valid JSON matching the requested schema without conversational filler."},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.1,
                        max_tokens=2048,
                        timeout=50.0
                    )
                    content = response.choices[0].message.content
                    if not content:
                        continue
                    cleaned = content.strip()
                    import re
                    match = re.search(r"\{[\s\S]*\}", cleaned)
                    if match:
                        try:
                            data = json.loads(match.group(0))
                            logger.info("Report generated successfully using NVIDIA NIM (%s, vision=%s)", m, use_vision)
                            return data
                        except Exception:
                            pass
                except Exception as inner_e:
                    logger.warning("NVIDIA model %s (vision=%s) failed: %s", m, use_vision, inner_e)
                    continue
            return None
        except Exception as e:
            logger.warning("NVIDIA report generation error: %s", e)
            return None
