from __future__ import annotations

import io
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.models import ScenarioPack


def generate_after_action_pdf(scenario: ScenarioPack, decisions: list[dict]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Crisis Exercise Pack + After-Action Review", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Scenario ID: {scenario.scenario_id}", styles["Normal"]))
    elements.append(Paragraph(f"Title: {scenario.title}", styles["Heading2"]))
    elements.append(Paragraph(scenario.overview, styles["BodyText"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Role Cards", styles["Heading2"]))
    for role, card in scenario.role_cards.model_dump().items():
        elements.append(Paragraph(f"<b>{role}</b>", styles["Heading3"]))
        elements.append(Paragraph("Responsibilities: " + "; ".join(card["responsibilities"]), styles["BodyText"]))
        elements.append(Paragraph("Priorities: " + "; ".join(card["priorities"]), styles["BodyText"]))
        elements.append(Paragraph("Do: " + "; ".join(card["dos"]), styles["BodyText"]))
        elements.append(Paragraph("Don't: " + "; ".join(card["donts"]), styles["BodyText"]))
        elements.append(Spacer(1, 8))

    elements.append(Paragraph("Timeline Phases", styles["Heading2"]))
    for phase in scenario.phases:
        elements.append(Paragraph(f"{phase.phase_id} ({phase.timeframe})", styles["Heading3"]))
        elements.append(Paragraph(phase.situation_summary, styles["BodyText"]))
        elements.append(Paragraph("Objectives: " + "; ".join(phase.objectives), styles["BodyText"]))
        elements.append(Spacer(1, 6))

    elements.append(Paragraph("Decisions Log", styles["Heading2"]))
    if not decisions:
        elements.append(Paragraph("No decisions submitted.", styles["BodyText"]))
    else:
        totals = []
        all_flags = []
        all_recommendations = []

        for idx, item in enumerate(decisions, start=1):
            eval_data = item.get("evaluation", {})
            totals.append(eval_data.get("total_score", 0))
            all_flags.extend(eval_data.get("risk_flags", []))
            all_recommendations.extend(eval_data.get("recommendations", []))

            elements.append(Paragraph(f"Decision {idx}", styles["Heading3"]))
            elements.append(Paragraph(f"Phase: {item.get('phase_id', '')}", styles["BodyText"]))
            elements.append(Paragraph(f"Question: {item.get('question', '')}", styles["BodyText"]))
            elements.append(Paragraph(f"Action: {item.get('decision_text', '')}", styles["BodyText"]))
            elements.append(Paragraph(f"Total Score: {eval_data.get('total_score', 0)}/25", styles["BodyText"]))
            elements.append(Spacer(1, 6))

        avg_total = sum(totals) / len(totals)
        elements.append(Paragraph("Score Summary", styles["Heading2"]))
        elements.append(Paragraph(f"Average total score: {avg_total:.2f}/25", styles["BodyText"]))

        elements.append(Paragraph("Key Risk Flags", styles["Heading2"]))
        counts = Counter(all_flags)
        if counts:
            for flag, count in counts.most_common(10):
                elements.append(Paragraph(f"- {flag} ({count})", styles["BodyText"]))
        else:
            elements.append(Paragraph("No recurring risk flags.", styles["BodyText"]))

        elements.append(Paragraph("Recommendations", styles["Heading2"]))
        if all_recommendations:
            deduped = []
            seen = set()
            for rec in all_recommendations:
                if rec not in seen:
                    seen.add(rec)
                    deduped.append(rec)
            for rec in deduped[:10]:
                elements.append(Paragraph(f"- {rec}", styles["BodyText"]))
        else:
            elements.append(Paragraph("No recommendations captured.", styles["BodyText"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()
