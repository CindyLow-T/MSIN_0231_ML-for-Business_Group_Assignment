# AI Interview Copilot (HireMind)

Streamlit MVP for AI-powered interview simulation, response evaluation, and personalized coaching.

## Core Features

1. Interview Setup
- Configure target role, company type/size/location, interview style, question count, and plan duration.
- Upload CV (PDF) or edit CV text directly.

2. Interview Room
- AI generates role-specific questions and adaptive follow-up questions.
- Supports both text mode and voice mode.
- Voice mode supports question read-aloud and response transcription.

3. Response Evaluation
- Per-question rubric scoring:
  - Overall
  - Structure
  - Clarity
  - Impact
  - STAR
- Includes strengths, weaknesses, improvement suggestions, and coaching feedback.

4. Performance Report
- Interview score trend chart across rounds.
- Interview history summary.
- Final synthesized report with personalized training plan.
- Report preview + PDF export.

## Project Structure

```text
MVP_Notebook.py
src/
  interview_models.py
  interview_service.py
  llm.py
prompts/
  interview_question_system.txt
  interview_question_user.txt
  interview_eval_system.txt
  interview_eval_user.txt
  interview_summary_system.txt
  interview_summary_user.txt
requirements.txt
.env.example
README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

```bash
cp .env.example .env
# Fill OPENAI_API_KEY
```

4. Run app:

```bash
streamlit run MVP_Notebook.py
```

## Notes

- This repository is for `MSIN_0231_ML-for-Business_Group_Assignment`.
- Main product entrypoint is `MVP_Notebook.py`.
