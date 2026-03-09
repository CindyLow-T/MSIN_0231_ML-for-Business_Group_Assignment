from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Type

from openai import OpenAI
from pydantic import BaseModel


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", max_retries: int = 2):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required")

        self.client = OpenAI(api_key=key)
        self.model = model
        self.max_retries = max_retries

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned empty response")
        return content.strip()

    @staticmethod
    def _extract_json_block(text: str) -> str:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.replace("```json", "").replace("```", "").strip()

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return candidate[start : end + 1]

    def _validate_json(self, text: str, schema_model: Type[BaseModel]) -> BaseModel:
        json_text = self._extract_json_block(text)
        payload = json.loads(json_text)
        return schema_model.model_validate(payload)

    def _repair_json(self, broken_output: str, schema_model: Type[BaseModel], error_message: str) -> str:
        system_prompt = load_prompt("json_repair_system.txt")
        user_template = load_prompt("json_repair_user.txt")
        user_prompt = user_template.format(
            schema=json.dumps(schema_model.model_json_schema(), ensure_ascii=False, indent=2),
            broken_json=broken_output,
            error_message=error_message,
        )
        return self._chat(system_prompt, user_prompt, temperature=0.0)

    def generate_validated_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[BaseModel],
        temperature: float,
    ) -> BaseModel:
        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            raw = self._chat(system_prompt, user_prompt, temperature=temperature)
            try:
                return self._validate_json(raw, schema_model)
            except Exception as parse_err:
                last_error = parse_err

            try:
                repaired = self._repair_json(raw, schema_model, str(last_error))
                return self._validate_json(repaired, schema_model)
            except Exception as repair_err:
                last_error = repair_err

        raise ValueError(f"Failed to produce valid JSON after retries: {last_error}")
