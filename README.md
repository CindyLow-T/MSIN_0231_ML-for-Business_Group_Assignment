# CrisisSim Studio

Production-style Streamlit MVP for enterprise tabletop crisis simulation.

## Features

1. Scenario Builder
- Generates a full structured crisis exercise pack via OpenAI.
- Uses strict JSON schema and Pydantic validation.
- Includes JSON repair retry flow when output is invalid.

2. Simulation Room
- Walk through phases with injects and decision points.
- Submit suggested or custom actions.
- Logs decisions in local SQLite.

3. Lightweight Decision Evaluation
- Rubric-based scoring (0-25) across legal, PR, customer, operations, investor dimensions.
- Returns reasons, risk flags, and recommendations.

4. Analytics
- Score trend line chart.
- Dimension breakdown bar chart.
- Decisions table.
- Recurring risk flags summary.

5. PDF Export
- Generates "Crisis Exercise Pack + After-Action Review" via reportlab.

## Project Structure

```text
app.py
src/
  models.py
  llm.py
  scenario.py
  evaluation.py
  storage.py
  report.py
prompts/
  scenario_system.txt
  scenario_user.txt
  eval_system.txt
  eval_user.txt
  json_repair_system.txt
  json_repair_user.txt
tests/
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
streamlit run app.py
```

## Run Tests

```bash
pytest -q
```

## Optional: Run AI Interview Copilot (Second App)

```bash
streamlit run interview_app.py
```

The interview app supports:
- Role/company-type/CV-based mock interviews
- Company profile presets (General, Google, Amazon, McKinsey)
- AI follow-up questions
- Per-answer scoring (Structure, Clarity, Impact, STAR)
- Improved sample answers + coaching tips
- Final strengths/weaknesses/suggested answers + personalized training plan

## Notes for Coursework

- Treat `app.py` as your product MVP.
- Build pitch deck and MVP-structure slides separately in PowerPoint/Canva, then export as PDFs.
- Include code zip from this repository in submission.
