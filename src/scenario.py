from __future__ import annotations

from src.llm import LLMClient, load_prompt
from src.models import CrisisInputs, ScenarioPack


class ScenarioService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(self, inputs: CrisisInputs) -> ScenarioPack:
        system_prompt = load_prompt("scenario_system.txt")
        user_template = load_prompt("scenario_user.txt")
        user_prompt = user_template.format(
            industry=inputs.industry,
            company_size=inputs.company_size,
            region=inputs.region,
            crisis_type=inputs.crisis_type,
            phase_count=inputs.phase_count,
            severity=inputs.severity,
        )

        result = self.llm_client.generate_validated_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=ScenarioPack,
            temperature=0.6,
        )
        return ScenarioPack.model_validate(result.model_dump())
