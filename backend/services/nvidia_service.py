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
                max_tokens=500,
                response_format={"type": "json_object"},
                timeout=60.0
            )
            content = response.choices[0].message.content
            if not content:
                logger.error("NVIDIA returned empty QA response.")
                return None
            logger.info("NVIDIA Raw Content: " + repr(content))

            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = chr(10).join(lines).strip()

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

    def generate_report(self, prompt: str) -> Optional[dict]:
        """Generate structured clinical report via NVIDIA NIM."""
        if not self.client:
            logger.error("NVIDIA client not initialized (missing API key).")
            return None
        try:
            models_to_try = [self.model, "meta/llama-3.2-11b-vision-instruct", "mistralai/mistral-large-2-instruct"]
            for m in models_to_try:
                try:
                    response = self.client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": "You are a clinical diagnostic reporting AI. Output strictly valid JSON matching the requested schema without conversational filler."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1500,
                        timeout=45.0
                    )
                    content = response.choices[0].message.content
                    if not content:
                        continue
                    cleaned = content.strip()
                    if "```" in cleaned:
                        parts = cleaned.split("```")
                        for p in parts:
                            p = p.strip()
                            if p.startswith("json"):
                                p = p[4:].strip()
                            try:
                                data = json.loads(p)
                                logger.info("Report generated successfully using NVIDIA NIM (%s)", m)
                                return data
                            except Exception:
                                continue
                    first_brace = cleaned.find("{")
                    last_brace = cleaned.rfind("}")
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        data = json.loads(cleaned[first_brace:last_brace+1])
                        logger.info("Report generated successfully using NVIDIA NIM (%s)", m)
                        return data
                except Exception as inner_e:
                    logger.warning("NVIDIA model %s failed: %s", m, inner_e)
                    continue
            return None
        except Exception as e:
            logger.warning("NVIDIA report generation error: %s", e)
            return None
