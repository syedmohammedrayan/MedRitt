import os
import json
import logging
import asyncio
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, ValidationError

from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

class ReportSchema(BaseModel):
    technique: str
    comparison: str
    image_quality: str
    findings: str
    impression: str
    differential_diagnosis: str
    recommendations: str
    critical_communication: str
    patient_explanation: str

class GeminiService:
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash"):
        self.api_key = api_key
        self.model = model or "gemini-3.5-flash"
        self.client = genai.Client(api_key=api_key)

    async def generate_structured_report(self, prompt: str, image_bytes: Optional[bytes] = None) -> Optional[ReportSchema]:
        if not self.api_key:
            logger.error("Gemini API key is not configured.")
            return None

        contents = []
        if image_bytes:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

        contents.append(prompt)

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReportSchema,
            temperature=0.1,
        )

        for attempt in range(2):
            try:
                task = self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                response = await asyncio.wait_for(task, timeout=60.0)

                if not response.text:
                    logger.error("Gemini returned empty response.")
                    return None

                try:
                    data = json.loads(response.text)
                    return ReportSchema(**data)
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"Malformed Gemini structured output: {e}")
                    return None

            except asyncio.TimeoutError:
                logger.warning(f"Gemini API timeout on attempt {attempt + 1}")
                if attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                break
            except APIError as e:
                logger.error(f"Gemini API Error: {e}")
                if any(err_code in str(e) for err_code in ["429", "503", "UNAVAILABLE"]):
                    await asyncio.sleep(2.0)
                    continue
                break
            except Exception as e:
                logger.error(f"Unexpected error calling Gemini: {e}")
                break

        return None

    async def generate_text(self, prompt: str) -> Optional[str]:
        if not self.api_key:
            logger.error("Gemini API key is not configured.")
            return None

        config = types.GenerateContentConfig(
            temperature=0.0,
        )
        for attempt in range(2):
            try:
                task = self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                response = await asyncio.wait_for(task, timeout=15.0)
                return response.text
            except asyncio.TimeoutError:
                logger.warning(f"Gemini text generation timeout on attempt {attempt + 1}")
            except APIError as e:
                logger.error(f"Gemini API Error: {e}")
                if "429" in str(e):
                    await asyncio.sleep(2.0)
                    continue
                break
            except Exception as e:
                logger.error(f"Error generating text with Gemini: {e}")
                break
        return None
