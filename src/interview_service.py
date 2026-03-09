from __future__ import annotations

import json

from src.interview_models import AnswerEvaluation, InterviewQuestion, InterviewSummary
from src.llm import LLMClient, load_prompt


class InterviewService:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_first_question(
        self,
        *,
        job_role: str,
        company_type: str,
        company_target: str,
        company_profile: str,
        cv_text: str,
        interview_style: str,
    ) -> InterviewQuestion:
        system_prompt = load_prompt("interview_question_system.txt")
        user_template = load_prompt("interview_question_user.txt")
        user_prompt = user_template.format(
            job_role=job_role,
            company_type=company_type,
            company_target=company_target,
            company_profile=company_profile,
            interview_style=interview_style,
            cv_text=cv_text[:5000],
            previous_question="",
            previous_answer="",
            round_index=1,
        )
        return self.llm.generate_validated_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=InterviewQuestion,
            temperature=0.4,
        )

    def evaluate_answer(
        self,
        *,
        job_role: str,
        company_type: str,
        company_target: str,
        company_profile: str,
        cv_text: str,
        interview_style: str,
        question: str,
        answer: str,
        round_index: int,
    ) -> AnswerEvaluation:
        system_prompt = load_prompt("interview_eval_system.txt")
        user_template = load_prompt("interview_eval_user.txt")
        user_prompt = user_template.format(
            job_role=job_role,
            company_type=company_type,
            company_target=company_target,
            company_profile=company_profile,
            interview_style=interview_style,
            cv_text=cv_text[:5000],
            question=question,
            answer=answer,
            round_index=round_index,
        )
        return self.llm.generate_validated_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=AnswerEvaluation,
            temperature=0.2,
        )

    def summarize_interview(
        self,
        *,
        job_role: str,
        company_type: str,
        company_target: str,
        company_profile: str,
        interview_style: str,
        plan_duration: str,
        qa_records: list[dict],
    ) -> InterviewSummary:
        system_prompt = load_prompt("interview_summary_system.txt")
        user_template = load_prompt("interview_summary_user.txt")
        user_prompt = user_template.format(
            job_role=job_role,
            company_type=company_type,
            company_target=company_target,
            company_profile=company_profile,
            interview_style=interview_style,
            plan_duration=plan_duration,
            qa_records=json.dumps(qa_records, ensure_ascii=False, indent=2),
        )
        return self.llm.generate_validated_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=InterviewSummary,
            temperature=0.2,
        )
