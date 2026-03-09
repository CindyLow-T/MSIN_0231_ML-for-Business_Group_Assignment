from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, conint


class InterviewQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    intent: str


class AnswerEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_score: conint(ge=0, le=10)
    structure_score: conint(ge=0, le=10)
    clarity_score: conint(ge=0, le=10)
    impact_score: conint(ge=0, le=10)
    star_score: conint(ge=0, le=10)
    strengths: list[str] = Field(min_length=1)
    weaknesses: list[str] = Field(min_length=1)
    improved_answer: str
    coach_tip: str
    follow_up_question: str


class InterviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_score: conint(ge=0, le=10)
    strengths: list[str] = Field(min_length=1)
    weaknesses: list[str] = Field(min_length=1)
    overall_improvement_suggestions: list[str] = Field(min_length=1)
    personalized_training_plan: list[str] = Field(min_length=1)
