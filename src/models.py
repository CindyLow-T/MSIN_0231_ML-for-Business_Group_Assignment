from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, conint


class CrisisInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    industry: Literal[
        "fintech",
        "retail",
        "healthcare",
        "tech",
        "manufacturing",
        "telecom",
        "energy",
        "logistics",
        "insurance",
        "public sector",
        "education",
        "travel & hospitality",
    ]
    company_size: Literal[
        "Startup (1-50)",
        "SME (51-250)",
        "Mid-Market (251-2000)",
        "Enterprise (2001-10000)",
        "Global Enterprise (10001+)",
    ]
    region: Literal["UK", "EU", "US", "Asia"]
    crisis_type: Literal[
        "PR scandal",
        "data breach",
        "supply chain disruption",
        "financial irregularity",
        "ransomware attack",
        "service outage",
        "regulatory investigation",
        "product safety recall",
        "executive misconduct",
        "insider misconduct",
        "AI model failure",
        "labor strike",
    ]
    phase_count: conint(ge=4, le=8) = 8
    severity: conint(ge=1, le=5)


class RoleCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    responsibilities: list[str]
    priorities: list[str]
    dos: list[str]
    donts: list[str]


class RoleCards(BaseModel):
    model_config = ConfigDict(extra="forbid")

    CEO: RoleCard
    PR: RoleCard
    Legal: RoleCard
    Ops: RoleCard
    CustomerSupport: RoleCard
    InvestorRelations: RoleCard


class Injects(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_article: str
    social_posts: list[str] = Field(min_length=5, max_length=5)
    regulator_message: str
    internal_email: str


class DecisionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    options: list[str] = Field(min_length=4, max_length=4)


class Phase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase_id: str
    timeframe: str
    situation_summary: str
    injects: Injects
    objectives: list[str]
    decision_points: list[DecisionPoint] = Field(min_length=1)


class Artifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    press_release_template: str
    internal_allhands_template: str
    customer_notice_template: str


class ScenarioPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    title: str
    overview: str
    assumptions: list[str]
    stakeholders: list[str]
    constraints: list[str]
    role_cards: RoleCards
    phases: list[Phase] = Field(min_length=2)
    artifacts: Artifacts


class EvaluationScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    legal_compliance: conint(ge=0, le=5)
    pr_reputation: conint(ge=0, le=5)
    customer_impact: conint(ge=0, le=5)
    operational_feasibility: conint(ge=0, le=5)
    investor_financial: conint(ge=0, le=5)


class DecisionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: EvaluationScores
    total_score: conint(ge=0, le=25)
    reasons: list[str]
    risk_flags: list[str]
    recommendations: list[str]
