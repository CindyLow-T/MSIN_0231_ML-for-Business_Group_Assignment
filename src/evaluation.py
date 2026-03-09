from __future__ import annotations

import json

from src.llm import LLMClient, load_prompt
from src.models import DecisionEvaluation, Phase, ScenarioPack


class EvaluationService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate_decision(
        self,
        *,
        scenario: ScenarioPack,
        phase: Phase,
        question: str,
        decision_text: str,
        constraints: str,
    ) -> DecisionEvaluation:
        system_prompt = load_prompt("eval_system.txt")
        user_template = load_prompt("eval_user.txt")

        user_prompt = user_template.format(
            scenario_id=scenario.scenario_id,
            scenario_title=scenario.title,
            crisis_overview=scenario.overview,
            phase_id=phase.phase_id,
            timeframe=phase.timeframe,
            situation_summary=phase.situation_summary,
            question=question,
            decision_text=decision_text,
            constraints=constraints or "None",
            objectives=json.dumps(phase.objectives, ensure_ascii=False),
            stakeholders=json.dumps(scenario.stakeholders, ensure_ascii=False),
        )

        result = self.llm_client.generate_validated_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=DecisionEvaluation,
            temperature=0.2,
        )
        return DecisionEvaluation.model_validate(result.model_dump())
