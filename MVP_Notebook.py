from __future__ import annotations

import base64
from datetime import datetime
import html
import io
import json
import os
import time

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.interview_service import InterviewService
from src.llm import LLMClient

load_dotenv()

# --------------------------------------------
# App bootstrap
# --------------------------------------------
# Configure the Streamlit app shell and sidebar default state.
st.set_page_config(
    page_title="AI Interview Copilot",
    layout="wide",
    initial_sidebar_state="collapsed" if st.session_state.get("iv_sidebar_collapsed", False) else "expanded",
)

# --------------------------------------------
# Session state defaults
# --------------------------------------------
# Keep all cross-page interview state in Streamlit session storage.
if "iv_api_key" not in st.session_state:
    st.session_state.iv_api_key = os.getenv("OPENAI_API_KEY", "")
if "iv_started" not in st.session_state:
    st.session_state.iv_started = False
if "iv_current_question" not in st.session_state:
    st.session_state.iv_current_question = ""
if "iv_current_intent" not in st.session_state:
    st.session_state.iv_current_intent = ""
if "iv_round" not in st.session_state:
    st.session_state.iv_round = 0
if "iv_records" not in st.session_state:
    st.session_state.iv_records = []
if "iv_summary" not in st.session_state:
    st.session_state.iv_summary = None
if "iv_setup" not in st.session_state:
    st.session_state.iv_setup = None
if "iv_intro_completed" not in st.session_state:
    st.session_state.iv_intro_completed = False
if "iv_sidebar_collapsed" not in st.session_state:
    st.session_state.iv_sidebar_collapsed = False
if "iv_api_saved_notice" not in st.session_state:
    st.session_state.iv_api_saved_notice = False
if "iv_cv_text" not in st.session_state:
    st.session_state.iv_cv_text = ""
if "iv_cv_file_sig" not in st.session_state:
    st.session_state.iv_cv_file_sig = ""
if "iv_cv_uploader_nonce" not in st.session_state:
    st.session_state.iv_cv_uploader_nonce = 0
if "iv_clear_cv_state" not in st.session_state:
    st.session_state.iv_clear_cv_state = False
if "iv_session_nonce" not in st.session_state:
    st.session_state.iv_session_nonce = 0
if "iv_nav_page" not in st.session_state:
    st.session_state.iv_nav_page = "Interview Setup"
if "iv_api_input" not in st.session_state:
    st.session_state.iv_api_input = ""
if "iv_clear_api_input" not in st.session_state:
    st.session_state.iv_clear_api_input = False
if "iv_model_name" not in st.session_state:
    st.session_state.iv_model_name = "gpt-4o-mini"
if "iv_api_status_level" not in st.session_state:
    st.session_state.iv_api_status_level = ""
if "iv_api_status_msg" not in st.session_state:
    st.session_state.iv_api_status_msg = ""
if "iv_question_audio_cache" not in st.session_state:
    st.session_state.iv_question_audio_cache = {}


def reset_interview() -> None:
    """Clear interview setup/results while keeping app-level preferences."""
    st.session_state.iv_started = False
    st.session_state.iv_current_question = ""
    st.session_state.iv_current_intent = ""
    st.session_state.iv_round = 0
    st.session_state.iv_records = []
    st.session_state.iv_summary = None
    st.session_state.iv_setup = None
    st.session_state.iv_clear_cv_state = True
    st.session_state.iv_cv_uploader_nonce += 1
    st.session_state.iv_session_nonce += 1
    st.session_state.iv_question_audio_cache = {}


def help_icon(help_text: str) -> str:
    """Render a small tooltip icon used in section headers."""
    safe = html.escape(help_text, quote=True)
    return f"<span class='iv-help' title='{safe}'>ⓘ</span>"


def is_api_key_format_valid(api_key: str) -> bool:
    """Perform lightweight local format validation before remote API check."""
    key = api_key.strip()
    return key.startswith("sk-") and len(key) >= 20


def validate_openai_api_key(api_key: str, model_name: str) -> tuple[bool, str]:
    """Validate key by attempting to retrieve the selected model."""
    try:
        client = OpenAI(api_key=api_key)
        client.models.retrieve(model_name)
        return True, ""
    except Exception as exc:  # pragma: no cover - external API behavior
        return False, str(exc)


def validate_api_input_on_enter() -> None:
    """Validate API key on Enter and update sidebar/session status flags."""
    entered_key = st.session_state.get("iv_api_input", "").strip()
    model_name = st.session_state.get("iv_model_name", "gpt-4o-mini")
    st.session_state.iv_api_status_level = ""
    st.session_state.iv_api_status_msg = ""

    if not entered_key:
        st.session_state.iv_api_key = ""
        st.session_state.iv_sidebar_collapsed = False
        st.session_state.iv_api_status_level = "warning"
        st.session_state.iv_api_status_msg = "Please enter your API key first."
        return

    if not is_api_key_format_valid(entered_key):
        st.session_state.iv_api_key = ""
        st.session_state.iv_sidebar_collapsed = False
        st.session_state.iv_api_status_level = "error"
        st.session_state.iv_api_status_msg = "Invalid key format. Please re-enter."
        return

    ok, err = validate_openai_api_key(entered_key, model_name)
    if ok:
        st.session_state.iv_api_key = entered_key
        st.session_state.iv_clear_api_input = True
        st.session_state.iv_sidebar_collapsed = True
        st.session_state.iv_api_saved_notice = True
        st.session_state.iv_api_status_level = "success"
        st.session_state.iv_api_status_msg = "API key validated."
    else:
        st.session_state.iv_api_key = ""
        st.session_state.iv_sidebar_collapsed = False
        st.session_state.iv_api_status_level = "error"
        st.session_state.iv_api_status_msg = "API key validation failed. Please re-enter."
        if err:
            st.session_state.iv_api_status_msg += f" ({err[:140]})"


def render_score_trend(records: list[dict]) -> None:
    """Render round-by-round overall score trend for completed answers."""
    if not records:
        return
    df = pd.DataFrame(records).copy()
    df["round"] = df["round"].astype(int)
    trend = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(
                "round:O",
                title="Interview Round",
                sort=sorted(df["round"].unique().tolist()),
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("overall_score:Q", title="Overall Score", scale=alt.Scale(domain=[0, 10])),
            tooltip=[
                alt.Tooltip("round:O", title="Round"),
                alt.Tooltip("overall_score:Q", title="Overall", format=".0f"),
                alt.Tooltip("structure_score:Q", title="Structure", format=".0f"),
                alt.Tooltip("clarity_score:Q", title="Clarity", format=".0f"),
                alt.Tooltip("impact_score:Q", title="Impact", format=".0f"),
                alt.Tooltip("star_score:Q", title="STAR", format=".0f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(trend, use_container_width=True)


def render_evaluation_details(record: dict) -> None:
    """Show rubric metrics and qualitative feedback for one response."""
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Overall", f"{record['overall_score']}/10")
    m2.metric("Structure", record["structure_score"])
    m3.metric("Clarity", record["clarity_score"])
    m4.metric("Impact", record["impact_score"])
    m5.metric("STAR", record["star_score"])

    st.markdown("**Strengths**")
    for s in record["strengths"]:
        st.write(f"- {s}")
    st.markdown("**Weaknesses**")
    for w in record["weaknesses"]:
        st.write(f"- {w}")
    st.markdown("**Improved Answer (Reference)**")
    st.write(record["improved_answer"])
    st.markdown("**Coach Tip**")
    st.write(record["coach_tip"])


def render_question_review(record: dict, *, expanded_eval: bool = False) -> None:
    """Render a question card with answer and expandable evaluation details."""
    st.markdown(f"### Question {record['round']}")
    st.caption(record["question"])
    with st.expander("Your Answer", expanded=False):
        st.write(record["answer"])
    with st.expander(f"Evaluation Score ({record['overall_score']}/10)", expanded=expanded_eval):
        render_evaluation_details(record)


def build_company_profile(company_type: str, company_size: str, company_location: str) -> str:
    """Create a compact company context string used in prompts."""
    return (
        f"{company_type} organization, company size: {company_size}, location: {company_location}. "
        "Prioritize structured communication, measurable impact, role fit, and practical execution."
    )


def generate_final_summary(setup: dict, llm: LLMClient | None) -> bool:
    """Call summary pipeline and store final report data in session state."""
    if llm is None:
        st.error("Please save an OpenAI API key first.")
        return False
    try:
        service = InterviewService(llm)
        with st.spinner("Generating final interview summary..."):
            summary = service.summarize_interview(
                job_role=setup.get("job_role", "Unknown role"),
                company_type=setup.get("company_type", "General Corporate"),
                company_target=f"{setup.get('company_size', 'Unknown size')} in {setup.get('company_location', 'Unknown location')}",
                company_profile=setup.get(
                    "company_profile",
                    build_company_profile(
                        setup.get("company_type", "General Corporate"),
                        setup.get("company_size", "Unknown size"),
                        setup.get("company_location", "Unknown location"),
                    ),
                ),
                interview_style=setup.get("interview_style", "Mixed"),
                plan_duration=setup.get("plan_duration", "Within 1 month"),
                qa_records=st.session_state.iv_records,
            )
        st.session_state.iv_summary = summary.model_dump()
        st.session_state.iv_started = False
        st.success("Interview performance report generated.")
        return True
    except Exception as exc:
        st.error(f"Summary generation failed: {exc}")
        return False


def build_report_preview(setup: dict, records: list[dict], summary: dict) -> str:
    """Build a plain-text report preview shown before PDF download."""
    improvement_items = summary.get("overall_improvement_suggestions") or summary.get("suggested_answers", [])
    lines = [
        "# AI Interview Copilot - Performance Report",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Job Role: {setup.get('job_role', 'N/A')}",
        f"- Company Type: {setup.get('company_type', 'N/A')}",
        f"- Interview Mode: {setup.get('interview_mode', 'N/A')}",
        f"- Company Size: {setup.get('company_size', 'N/A')}",
        f"- Company Location: {setup.get('company_location', 'N/A')}",
        f"- Interview Style: {setup.get('interview_style', 'N/A')}",
        f"- Plan Duration: {setup.get('plan_duration', 'N/A')}",
        f"- Questions Completed: {len(records)}",
        "",
        "## Overall",
        f"- Overall Score: {summary.get('overall_score', 0)}/10",
        "",
        "## Strengths",
    ]
    for item in summary.get("strengths", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Weaknesses")
    for item in summary.get("weaknesses", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Overall Improvement Suggestions")
    for item in improvement_items:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Personalized Training Plan")
    for item in summary.get("personalized_training_plan", []):
        lines.append(f"- {item}")
    return "\n".join(lines)


def build_report_pdf(setup: dict, records: list[dict], summary: dict) -> bytes:
    """Generate the final downloadable performance report in PDF format."""
    improvement_items = summary.get("overall_improvement_suggestions") or summary.get("suggested_answers", [])

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    story = [
        Paragraph("AI Interview Copilot - Performance Report", title_style),
        Spacer(1, 12),
        Paragraph(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body),
        Spacer(1, 8),
    ]

    meta_rows = [
        ["Job Role", setup.get("job_role", "N/A")],
        ["Company Type", setup.get("company_type", "N/A")],
        ["Interview Mode", setup.get("interview_mode", "N/A")],
        ["Company Size", setup.get("company_size", "N/A")],
        ["Company Location", setup.get("company_location", "N/A")],
        ["Interview Style", setup.get("interview_style", "N/A")],
        ["Plan Duration", setup.get("plan_duration", "N/A")],
        ["Questions Completed", str(len(records))],
        ["Overall Score", f"{summary.get('overall_score', 0)}/10"],
    ]
    meta_table = Table(meta_rows, colWidths=[170, 340])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef5ff")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2937")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([meta_table, Spacer(1, 14)])

    def add_bullets(title: str, items: list[str]) -> None:
        story.append(Paragraph(title, h2))
        bullet_items = [ListItem(Paragraph(str(item), body), leftIndent=12) for item in items if str(item).strip()]
        if bullet_items:
            story.append(ListFlowable(bullet_items, bulletType="bullet", start="circle"))
        else:
            story.append(Paragraph("No items.", body))
        story.append(Spacer(1, 10))

    add_bullets("Strengths", summary.get("strengths", []))
    add_bullets("Weaknesses", summary.get("weaknesses", []))
    add_bullets("Overall Improvement Suggestions", improvement_items)
    add_bullets("Personalized Training Plan", summary.get("personalized_training_plan", []))

    story.append(Paragraph("Interview History Details", h2))
    for r in records:
        q_no = int(r.get("round", 0))
        q_text = html.escape(str(r.get("question", ""))).replace("\n", "<br/>")
        a_text = html.escape(str(r.get("answer", ""))).replace("\n", "<br/>")
        score_line = (
            f"Overall {int(r.get('overall_score', 0))}/10 | "
            f"Structure {int(r.get('structure_score', 0))} | "
            f"Clarity {int(r.get('clarity_score', 0))} | "
            f"Impact {int(r.get('impact_score', 0))} | "
            f"STAR {int(r.get('star_score', 0))}"
        )

        story.append(Paragraph(f"Question {q_no}", styles["Heading3"]))
        story.append(Paragraph(f"<b>Scores:</b> {score_line}", body))
        story.append(Spacer(1, 3))
        story.append(Paragraph(f"<b>Question:</b> {q_text}", body))
        story.append(Spacer(1, 3))
        story.append(Paragraph(f"<b>Your Answer:</b> {a_text}", body))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buffer.getvalue()


# Ordered pages in the main user workflow.
WORKFLOW_STEPS = ["Interview Setup", "Interview Room", "Response Evaluation", "Performance Report"]


def get_max_unlocked_step_idx() -> int:
    """Return the highest page index unlocked by current progress."""
    max_idx = 0
    if st.session_state.iv_started or st.session_state.iv_records:
        max_idx = 1
    if st.session_state.iv_records:
        max_idx = 3
    return max_idx


def go_to_step(step_idx: int) -> None:
    """Navigate to a workflow step while enforcing unlock constraints."""
    safe_idx = max(0, min(step_idx, get_max_unlocked_step_idx()))
    st.session_state.iv_nav_page = WORKFLOW_STEPS[safe_idx]
    st.rerun()


def render_progress_nav(current_idx: int, max_unlocked_idx: int) -> None:
    """Render top workflow navigation with lock/unlock behavior."""
    widths: list[int] = []
    for idx in range(len(WORKFLOW_STEPS)):
        widths.append(3)
        if idx < len(WORKFLOW_STEPS) - 1:
            widths.append(1)
    cols = st.columns(widths, gap="small")

    clicked_idx: int | None = None
    for idx, step_name in enumerate(WORKFLOW_STEPS):
        state = "current" if idx == current_idx else ("done" if idx < current_idx else "locked")
        with cols[idx * 2]:
            if st.button(
                f"{idx + 1}. {step_name}",
                key=f"iv_progress_btn_{idx}",
                disabled=idx > max_unlocked_idx,
                use_container_width=True,
                type="primary" if idx == current_idx else "tertiary",
            ):
                clicked_idx = idx

        if idx < len(WORKFLOW_STEPS) - 1:
            line_state = "done" if idx < current_idx else "todo"
            with cols[idx * 2 + 1]:
                st.markdown(f"<div class='p-line {line_state}'></div>", unsafe_allow_html=True)

    if clicked_idx is not None:
        go_to_step(clicked_idx)


def parse_pdf_text(uploaded_file) -> str:
    """Extract text from uploaded PDF CV for question personalization."""
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("PDF parsing package missing. Please install pypdf.") from exc

    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    parsed = "\n".join([p for p in pages if p]).strip()
    if not parsed:
        raise ValueError("No readable text found in the uploaded PDF.")
    return parsed


def transcribe_audio_answer(uploaded_audio, llm: LLMClient | None) -> str:
    """Transcribe recorded/uploaded audio answer into editable text."""
    if llm is None:
        raise ValueError("OpenAI API key is required for voice transcription.")
    if uploaded_audio is None:
        raise ValueError("Please record or upload an audio answer first.")

    audio_bytes = uploaded_audio.getvalue()
    if not audio_bytes:
        raise ValueError("Audio file is empty. Please record again.")

    filename = getattr(uploaded_audio, "name", "voice_answer.wav")
    last_error: Exception | None = None
    for model in ("gpt-4o-mini-transcribe", "whisper-1"):
        try:
            stream = io.BytesIO(audio_bytes)
            stream.name = filename
            result = llm.client.audio.transcriptions.create(model=model, file=stream)
            text = getattr(result, "text", "").strip()
            if text:
                return text
        except Exception as exc:  # pragma: no cover - external API behavior
            last_error = exc
    raise RuntimeError(f"Voice transcription failed: {last_error}")


def detect_audio_mime(data: bytes) -> str | None:
    """Infer audio MIME type from binary header bytes."""
    if not data:
        return None
    if data.startswith(b"ID3") or data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "audio/mpeg"
    if data.startswith(b"RIFF") and len(data) > 12 and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "audio/mp4"
    return None


def synthesize_question_audio(
    question: str,
    llm: LLMClient | None,
    *,
    voice: str = "nova",
    speed: float = 1.0,
) -> tuple[bytes, str]:
    """Synthesize spoken question audio, preferring higher-quality models."""
    if llm is None:
        raise ValueError("OpenAI API key is required for question voice playback.")
    prompt = question.strip()
    if not prompt:
        raise ValueError("No question available for voice playback.")

    last_error: Exception | None = None
    for model in ("tts-1-hd", "tts-1", "gpt-4o-mini-tts"):
        try:
            response = llm.client.audio.speech.create(
                model=model,
                voice=voice,
                input=prompt,
                response_format="wav",
                speed=speed,
            )
            if hasattr(response, "read"):
                data = response.read()
            elif hasattr(response, "content"):
                data = response.content
            elif hasattr(response, "iter_bytes"):
                data = b"".join(response.iter_bytes())
            else:
                data = bytes(response)
            if not data:
                continue
            mime = detect_audio_mime(data)
            if mime:
                return data, mime
        except Exception as exc:  # pragma: no cover - external API behavior
            last_error = exc
    raise RuntimeError(f"Question voice playback failed: {last_error}")


def speak_question_in_browser(question: str) -> None:
    """Local browser speech fallback using Web Speech API."""
    speak_text = json.dumps(question.strip())
    components.html(
        f"""
        <script>
          const text = {speak_text};

          function pickNaturalVoice(voices) {{
            if (!voices || !voices.length) return null;
            const qualityHints = [
              /siri/i,
              /neural/i,
              /enhanced/i,
              /premium/i,
              /natural/i,
              /google\\s*uk\\s*english/i,
              /microsoft.*aria/i,
              /microsoft.*jenny/i,
              /samantha/i,
              /karen/i,
              /daniel/i,
              /alex/i,
              /victoria/i,
            ];
            const langPriority = ["en-GB", "en-US", "en-AU", "en-CA", "en"];

            const ranked = voices.map((voice) => {{
              const name = (voice.name || "").toLowerCase();
              const lang = (voice.lang || "").toLowerCase();
              let score = 0;

              for (const re of qualityHints) {{
                if (re.test(name)) {{
                  score += 6;
                }}
              }}

              const idx = langPriority.findIndex((lp) => lang.startsWith(lp.toLowerCase()));
              if (idx >= 0) {{
                score += (langPriority.length - idx) * 2;
              }}

              if (voice.localService) score += 1;
              if (voice.default) score += 1;
              return {{ voice, score }};
            }});

            ranked.sort((a, b) => b.score - a.score);
            return ranked[0] ? ranked[0].voice : null;
          }}

          function speakWithBestVoice(rawText) {{
            if (!('speechSynthesis' in window) || !rawText) return;
            const synth = window.speechSynthesis;

            function doSpeak() {{
              const voices = synth.getVoices() || [];
              const best = pickNaturalVoice(voices);
              const utter = new SpeechSynthesisUtterance(rawText);
              if (best) {{
                utter.voice = best;
                utter.lang = best.lang || 'en-US';
              }} else {{
                utter.lang = 'en-US';
              }}
              utter.rate = 0.98;
              utter.pitch = 1.0;
              synth.cancel();
              synth.speak(utter);
            }}

            const voicesNow = synth.getVoices();
            if (voicesNow && voicesNow.length) {{
              doSpeak();
              return;
            }}

            let fired = false;
            const runOnce = () => {{
              if (fired) return;
              fired = true;
              synth.onvoiceschanged = null;
              doSpeak();
            }};
            synth.onvoiceschanged = runOnce;
            setTimeout(runOnce, 250);
          }}

          speakWithBestVoice(text);
        </script>
        """,
        height=0,
    )


def render_instant_read_button(question: str, round_id: int, cloud_audio_src: str = "") -> None:
    """Render single-click 'Read Question' with cloud-audio-first fallback logic."""
    safe_text = json.dumps(question.strip())
    safe_audio_src = json.dumps(cloud_audio_src)
    components.html(
        f"""
        <div style="display:flex; justify-content:center;">
          <button id="iv-read-btn"
            style="
              width:100%;
              max-width:760px;
              background:#1e3a8a;
              color:#fff;
              border:1px solid #1e3a8a;
              border-radius:10px;
              font-size:1rem;
              font-weight:600;
              line-height:1.2;
              padding:0.62rem 0.75rem;
              cursor:pointer;
            ">
            Read Question {round_id}
          </button>
        </div>
        <script>
          const text = {safe_text};
          const cloudAudioSrc = {safe_audio_src};

          function pickNaturalVoice(voices) {{
            if (!voices || !voices.length) return null;
            const qualityHints = [
              /siri/i,
              /neural/i,
              /enhanced/i,
              /premium/i,
              /natural/i,
              /samantha/i,
              /karen/i,
              /victoria/i,
              /google/i,
              /microsoft.*jenny/i,
              /microsoft.*aria/i,
            ];
            const langPriority = ["en-GB", "en-US", "en-AU", "en-CA", "en"];
            const ranked = voices.map((voice) => {{
              const name = (voice.name || "").toLowerCase();
              const lang = (voice.lang || "").toLowerCase();
              let score = 0;
              for (const re of qualityHints) {{
                if (re.test(name)) score += 6;
              }}
              const idx = langPriority.findIndex((lp) => lang.startsWith(lp.toLowerCase()));
              if (idx >= 0) score += (langPriority.length - idx) * 2;
              if (voice.localService) score += 1;
              if (voice.default) score += 1;
              return {{ voice, score }};
            }});
            ranked.sort((a, b) => b.score - a.score);
            return ranked[0] ? ranked[0].voice : null;
          }}

          function speakNow() {{
            if (!('speechSynthesis' in window) || !text) return;
            const synth = window.speechSynthesis;
            const voices = synth.getVoices() || [];
            const best = pickNaturalVoice(voices);
            const utter = new SpeechSynthesisUtterance(text);
            if (best) {{
              utter.voice = best;
              utter.lang = best.lang || "en-US";
            }} else {{
              utter.lang = "en-US";
            }}
            utter.rate = 0.98;
            utter.pitch = 1.0;
            synth.cancel();
            synth.speak(utter);
          }}

          const btn = document.getElementById("iv-read-btn");
          if (btn) {{
            btn.addEventListener("click", () => {{
              try {{
                if (window.__ivActiveAudio) {{
                  window.__ivActiveAudio.pause();
                  window.__ivActiveAudio.currentTime = 0;
                }}
              }} catch (e) {{}}
              try {{
                if ('speechSynthesis' in window) {{
                  window.speechSynthesis.cancel();
                }}
              }} catch (e) {{}}

              if (cloudAudioSrc) {{
                const audio = new Audio(cloudAudioSrc);
                audio.preload = "auto";
                audio.currentTime = 0;
                window.__ivActiveAudio = audio;
                audio.play().catch(() => {{
                  speakNow();
                }});
              }} else {{
                speakNow();
              }}
            }});
          }}
        </script>
        """,
        height=70,
    )


def autoplay_audio_bytes(
    audio_bytes: bytes, mime: str = "audio/mpeg", fallback_text: str = "", nonce: str | None = None
) -> None:
    """Autoplay helper for generated audio bytes with local speech fallback."""
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    fallback_json = json.dumps(fallback_text)
    run_nonce = nonce or str(time.time_ns())
    components.html(
        f"""
        <script>
          const runNonce = "{run_nonce}";
          function pickNaturalVoice(voices) {{
            if (!voices || !voices.length) return null;
            const qualityHints = [
              /siri/i,
              /neural/i,
              /enhanced/i,
              /premium/i,
              /natural/i,
              /google\\s*uk\\s*english/i,
              /microsoft.*aria/i,
              /microsoft.*jenny/i,
              /samantha/i,
              /karen/i,
              /daniel/i,
              /alex/i,
              /victoria/i,
            ];
            const langPriority = ["en-GB", "en-US", "en-AU", "en-CA", "en"];

            const ranked = voices.map((voice) => {{
              const name = (voice.name || "").toLowerCase();
              const lang = (voice.lang || "").toLowerCase();
              let score = 0;
              for (const re of qualityHints) {{
                if (re.test(name)) score += 6;
              }}
              const idx = langPriority.findIndex((lp) => lang.startsWith(lp.toLowerCase()));
              if (idx >= 0) score += (langPriority.length - idx) * 2;
              if (voice.localService) score += 1;
              if (voice.default) score += 1;
              return {{ voice, score }};
            }});
            ranked.sort((a, b) => b.score - a.score);
            return ranked[0] ? ranked[0].voice : null;
          }}

          function speakWithBestVoice(rawText) {{
            if (!('speechSynthesis' in window) || !rawText) return;
            const synth = window.speechSynthesis;
            function doSpeak() {{
              const voices = synth.getVoices() || [];
              const best = pickNaturalVoice(voices);
              const utter = new SpeechSynthesisUtterance(rawText);
              if (best) {{
                utter.voice = best;
                utter.lang = best.lang || "en-US";
              }} else {{
                utter.lang = "en-US";
              }}
              utter.rate = 0.98;
              utter.pitch = 1.0;
              synth.cancel();
              synth.speak(utter);
            }}
            const voicesNow = synth.getVoices();
            if (voicesNow && voicesNow.length) {{
              doSpeak();
              return;
            }}
            let fired = false;
            const runOnce = () => {{
              if (fired) return;
              fired = true;
              synth.onvoiceschanged = null;
              doSpeak();
            }};
            synth.onvoiceschanged = runOnce;
            setTimeout(runOnce, 250);
          }}

          const src = "data:{mime};base64,{encoded}";
          const audio = new Audio(src);
          audio.preload = "auto";
          audio.setAttribute("data-run-nonce", runNonce);
          let started = false;
          audio.addEventListener("playing", () => {{
            started = true;
          }});
          const tryPlay = () => audio.play().catch(() => null);
          tryPlay();
          audio.addEventListener("canplaythrough", () => {{
            tryPlay();
          }});
          setTimeout(() => {{
            tryPlay();
            if (!started) {{
              const txt = {fallback_json};
              speakWithBestVoice(txt);
            }}
          }}, 220);
        </script>
        """,
        height=0,
    )


# --------------------------------------------
# Global UI styling
# --------------------------------------------
# Centralized CSS tokens and component styling for consistent UX.
st.markdown(
    """
    <style>
      .main .block-container {
        padding-top: 0.65rem;
      }

      section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.55rem;
      }

      :root {
        --iv-blue-900: #1e3a8a;
        --iv-blue-700: #1d4ed8;
        --iv-blue-600: #2563eb;
        --iv-blue-200: #bfdbfe;
        --iv-blue-100: #dbeafe;
        --iv-blue-050: #eff6ff;
      }

      div[data-testid="stButton"] > button {
        border-radius: 10px;
        font-weight: 600;
      }

      div[data-testid="stButton"] button[kind="primary"],
      .stButton button[kind="primary"],
      button[data-testid="baseButton-primary"] {
        background: var(--iv-blue-900) !important;
        color: #ffffff !important;
        border: 1px solid var(--iv-blue-900) !important;
      }

      div[data-testid="stButton"] button[kind="primary"]:hover,
      .stButton button[kind="primary"]:hover,
      button[data-testid="baseButton-primary"]:hover {
        background: #172554 !important;
        color: #ffffff !important;
        border-color: #172554 !important;
      }

      div[data-testid="stButton"] > button[kind="secondary"] {
        background: var(--iv-blue-050);
        color: var(--iv-blue-900);
        border: 1px solid var(--iv-blue-200);
      }

      div[data-testid="stDownloadButton"] > button {
        border-radius: 10px;
        font-weight: 600;
        background: var(--iv-blue-900);
        color: #ffffff;
        border: 1px solid var(--iv-blue-900);
      }

      div[data-testid="stDownloadButton"] > button:hover {
        background: #172554;
        border-color: #172554;
      }

      div[data-testid="stButton"] > button[kind="tertiary"] {
        background: transparent;
        color: #1d4ed8;
        border: none;
        box-shadow: none;
        padding: 0.1rem 0.15rem;
        min-height: 1.2rem;
      }

      .intro-hero {
        background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
        border: 1px solid var(--iv-blue-200);
        border-radius: 14px;
        padding: 16px 18px;
      }

      .hero-eyebrow {
        margin: 0 0 8px 0;
        color: var(--iv-blue-700);
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.82rem;
      }

      .intro-hero h3 {
        margin: 0 0 8px 0;
        color: #0f172a;
        font-size: 1.62rem;
        line-height: 1.24;
      }

      .intro-hero p {
        margin: 0;
        color: #334155;
        font-size: 1rem;
      }

      .team-note {
        margin-top: 8px;
        color: #334155;
        font-size: 0.92rem;
      }

      .hero-chip-row {
        margin-top: 12px;
        display: flex;
        gap: 8px;
        flex-wrap: nowrap;
        overflow-x: auto;
        white-space: nowrap;
        scrollbar-width: thin;
      }

      .hero-chip {
        display: inline-flex;
        align-items: center;
        flex: 0 0 auto;
        padding: 5px 11px;
        border-radius: 999px;
        border: 1px solid #bfd6ff;
        background: #f5f9ff;
        color: #1f3f86;
        font-size: 0.84rem;
        font-weight: 600;
      }

      .section-title {
        margin: 0 0 8px 0;
        color: #0f172a;
        font-size: 1.85rem;
      }

      .workflow-wrap {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 12px;
      }

      .workflow-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }

      .workflow-card {
        background: #ffffff;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 12px 13px;
        min-height: 148px;
      }

      .workflow-card .w-step {
        color: var(--iv-blue-900);
        font-weight: 700;
        margin-bottom: 7px;
        font-size: 0.86rem;
        letter-spacing: 0.06em;
      }

      .workflow-card .w-name {
        color: #0f172a;
        font-size: 1.2rem;
        line-height: 1.2;
        font-weight: 700;
        margin-bottom: 8px;
      }

      .workflow-card .w-text {
        color: #1e3a8a;
        font-size: 0.93rem;
        line-height: 1.35;
      }

      .intro-section {
        margin-top: 14px;
      }

      .ready-wrap {
        background: #f7faff;
        border: 1px solid #d4e4ff;
        border-radius: 12px;
        padding: 14px;
      }

      .ready-title {
        margin: 0 0 6px 0;
        color: #0f172a;
        font-size: 1.9rem;
      }

      .ready-text {
        margin: 0;
        color: #334155;
        font-size: 1.04rem;
      }

      .app-title {
        font-size: 2.05rem;
        line-height: 1.12;
        margin: 0 0 0.15rem 0;
        color: #0f172a;
      }

      .app-members {
        margin: 0.1rem 0 0.45rem 0;
        color: #334155;
        font-size: 0.94rem;
      }

      .top-members {
        position: fixed;
        top: 0.82rem;
        right: 1.15rem;
        z-index: 1200;
        color: #334155;
        font-size: 0.94rem;
        line-height: 1.35;
        text-align: right;
        background: rgba(255, 255, 255, 0.9);
        padding: 0.1rem 0.15rem;
        border-radius: 6px;
      }

      .p-line {
        width: 100%;
        height: 1px;
        background: #cbd5e1;
        margin-top: 1.25rem;
      }

      .p-line.done {
        background: #60a5fa;
      }

      .action-hint {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 0.25rem;
      }

      .iv-help {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.05rem;
        height: 1.05rem;
        margin-left: 0.32rem;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        cursor: help;
        vertical-align: middle;
      }

      button[aria-label="Clear Transcript"] {
        color: #94a3b8 !important;
      }

      [data-testid="stDataFrame"] th {
        white-space: nowrap !important;
      }

      @media (max-width: 900px) {
        .workflow-grid {
          grid-template-columns: 1fr;
        }
        .workflow-card {
          min-height: 0;
        }
        .p-wrap {
          overflow-x: auto;
          padding-bottom: 3px;
        }
        .ready-title {
          font-size: 1.55rem;
        }
        .app-title {
          font-size: 1.9rem;
        }
        .hero-chip-row {
          flex-wrap: wrap;
          overflow-x: visible;
          white-space: normal;
        }
        .top-members {
          position: static;
          background: transparent;
          padding: 0;
          margin: 0 0 0.3rem 0;
          text-align: left;
        }
      }

      @media (max-width: 1450px) {
        .hero-chip-row {
          flex-wrap: wrap;
          overflow-x: visible;
          white-space: normal;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
# --------------------------------------------
# Sidebar controls
# --------------------------------------------
# API key input/validation and model selection.
)
st.sidebar.header("API Settings")
model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini"], index=0, key="iv_model_name")

if st.session_state.iv_clear_api_input:
    st.session_state.iv_api_input = ""
    st.session_state.iv_clear_api_input = False

api_input = st.sidebar.text_input(
    "OpenAI API Key",
    key="iv_api_input",
    type="password",
    placeholder="Paste key and press Enter",
    help="Saved in this browser session only.",
    on_change=validate_api_input_on_enter,
)
entered_key = api_input.strip()

if entered_key:
    if is_api_key_format_valid(entered_key):
        st.sidebar.caption("Key format looks correct. Press Enter to validate and save.")
    else:
        st.sidebar.error("This does not look like a valid OpenAI API key format. Please re-enter.")

if st.session_state.iv_api_saved_notice:
    st.sidebar.success("API key validated and saved for this session.")
    st.session_state.iv_api_saved_notice = False

if st.session_state.iv_api_status_msg:
    level = st.session_state.iv_api_status_level
    if level == "error":
        st.sidebar.error(st.session_state.iv_api_status_msg)
    elif level == "warning":
        st.sidebar.warning(st.session_state.iv_api_status_msg)
    elif level == "success":
        st.sidebar.success(st.session_state.iv_api_status_msg)
    else:
        st.sidebar.info(st.session_state.iv_api_status_msg)

if st.session_state.iv_api_key:
    st.sidebar.info("API key is loaded for this session.")
    if st.sidebar.button("Clear Saved API Key", use_container_width=True):
        st.session_state.iv_api_key = ""
        st.session_state.iv_clear_api_input = True
        st.session_state.iv_sidebar_collapsed = False
        st.rerun()
else:
    st.sidebar.warning("Please enter and validate a valid API key to continue.")

llm = None
if st.session_state.iv_api_key:
    llm = LLMClient(api_key=st.session_state.iv_api_key, model=model_name, max_retries=2)

# --------------------------------------------
# Global header + team details
# --------------------------------------------
st.markdown("<h1 class='app-title'>AI Interview Copilot</h1>", unsafe_allow_html=True)
st.caption("AI-powered interview coaching platform for realistic simulation, structured evaluation, and personalized improvement planning.")
st.markdown(
    "<div class='top-members'>Fui Man Triciacindy Low (25171581)<br/>Lewen Yang (25087575)</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------
# Intro/Landing page (pre-workflow)
# --------------------------------------------
if not st.session_state.iv_intro_completed:
    st.markdown(
        """
        <div class="intro-hero">
          <div class="hero-eyebrow">Interview Training Platform</div>
          <h3>Train like a real interview, with AI coaching after every answer.</h3>
          <p>Build confidence through realistic question flow, follow-up pressure, and actionable feedback.</p>
          <div class="hero-chip-row">
            <span class="hero-chip">Tailored Questions</span>
            <span class="hero-chip">Response Evaluation</span>
            <span class="hero-chip">Performance Analytics</span>
            <span class="hero-chip">Coaching Tips</span>
            <span class="hero-chip">Text Mode</span>
            <span class="hero-chip">Voice Mode</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='intro-section'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>Product Workflow</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workflow-wrap">
          <div class="workflow-grid">
            <div class="workflow-card">
              <div class="w-step">STEP 1</div>
              <div class="w-name">Interview Setup</div>
              <div class="w-text">Define role, company context, and CV details to start a realistic simulation.</div>
            </div>
            <div class="workflow-card">
              <div class="w-step">STEP 2</div>
              <div class="w-name">Interview Room</div>
              <div class="w-text">Answer adaptive interview questions one by one in a realistic flow.</div>
            </div>
            <div class="workflow-card">
              <div class="w-step">STEP 3</div>
              <div class="w-name">Response Evaluation</div>
              <div class="w-text">Review each answer with structured scoring, strengths, and coaching feedback.</div>
            </div>
            <div class="workflow-card">
              <div class="w-step">STEP 4</div>
              <div class="w-name">Performance Report</div>
              <div class="w-text">View trend analytics and export a final improvement report.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='intro-section'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ready-wrap">
          <h3 class="ready-title">Ready to begin?</h3>
          <p class="ready-text">Start Interview Setup to launch your first AI-led interview simulation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='intro-section'></div>", unsafe_allow_html=True)
    bottom_spacer, bottom_btn = st.columns([5, 2])
    with bottom_btn:
        if st.button("Start Interview Setup", type="primary", use_container_width=True, key="intro_start_bottom"):
            st.session_state.iv_intro_completed = True
            st.session_state.iv_nav_page = "Interview Setup"
            st.rerun()
    st.stop()

max_unlocked_idx = get_max_unlocked_step_idx()

if st.session_state.iv_nav_page not in WORKFLOW_STEPS:
    st.session_state.iv_nav_page = WORKFLOW_STEPS[0]
if WORKFLOW_STEPS.index(st.session_state.iv_nav_page) > max_unlocked_idx:
    st.session_state.iv_nav_page = WORKFLOW_STEPS[max_unlocked_idx]

current_idx = WORKFLOW_STEPS.index(st.session_state.iv_nav_page)
render_progress_nav(current_idx=current_idx, max_unlocked_idx=max_unlocked_idx)

st.markdown("---")
current_page = st.session_state.iv_nav_page

# ============================================
# Page 1: Interview Setup
# ============================================
if current_page == "Interview Setup":
    st.markdown(
        f"## Interview Setup {help_icon('Configure your role and interview context. These settings shape question style and final training plan.')}",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        # Subsection: Core interview context
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            job_role = st.selectbox(
                "Target Job Role",
                [
                    "Please select",
                    "Data Analyst",
                    "Business Analyst",
                    "Product Manager",
                    "Software Engineer",
                    "Data Scientist",
                    "Marketing Analyst",
                    "Operations Analyst",
                    "Consultant",
                ],
                index=0,
                key="iv_setup_job_role",
                help="Choose the role you are preparing for.",
            )
        with row1_col2:
            company_type = st.selectbox(
                "Company Type",
                [
                    "Please select",
                    "Consulting",
                    "Big Tech",
                    "Finance and Banking",
                    "Healthcare",
                    "Retail and E-commerce",
                    "Manufacturing",
                    "Public Sector",
                    "Telecommunications",
                    "Energy and Utilities",
                    "Media and Entertainment",
                    "Education",
                    "Nonprofit",
                    "Startup",
                    "General Corporate",
                ],
                index=0,
                key="iv_setup_company_type",
                help="Select the target company category for realistic interview context.",
            )
        with row1_col3:
            max_rounds = st.slider(
                "Number of Questions",
                min_value=1,
                max_value=10,
                value=5,
                key="iv_setup_max_rounds",
                help="Set how many interview rounds to run in this session.",
            )

        # Subsection: Company details + interview style
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            company_size = st.selectbox(
                "Company Size",
                [
                    "Please select",
                    "Pre-seed Startup (1-10)",
                    "Early-stage Startup (11-50)",
                    "Growth-stage Startup (51-200)",
                    "Scale-up (201-500)",
                    "Mid-market (501-2000)",
                    "Enterprise (2001-10000)",
                    "Global Enterprise (10000+)",
                ],
                index=0,
                key="iv_setup_company_size",
                help="Select company size to tune expectations and evaluation standards.",
            )
        with row2_col2:
            company_location = st.selectbox(
                "Company Location",
                [
                    "Please select",
                    "United Kingdom",
                    "Europe",
                    "United States",
                    "Asia-Pacific",
                    "Middle East",
                    "Global/Remote",
                ],
                index=0,
                key="iv_setup_company_location",
                help="Select geography to reflect local business context.",
            )
        with row2_col3:
            interview_style = st.selectbox(
                "Interview Style",
                ["Please select", "Mixed", "Behavioral", "Technical", "Case-based"],
                index=0,
                key="iv_setup_interview_style",
                help="Choose the interview focus style.",
            )

        # Subsection: Timeline + mode
        row3_col1, row3_col2, row3_col3 = st.columns(3)
        with row3_col1:
            plan_duration = st.selectbox(
                "Plan Duration",
                [
                    "Please select",
                    "Interview in 1 week",
                    "Interview in 2 weeks",
                    "Interview in 1 month",
                    "Interview in 2-3 months",
                    "Flexible timeline",
                ],
                index=0,
                key="iv_setup_plan_duration",
                help="How soon your interview is coming. Used to personalize the training plan. Tip: choose a shorter timeline for a more intensive plan.",
            )
        with row3_col2:
            interview_mode = st.selectbox(
                "Interview Mode",
                ["Please select", "Text", "Voice"],
                index=0,
                key="iv_setup_interview_mode",
                help="Text mode uses typing only. Voice mode lets you record and transcribe answers.",
            )
        with row3_col3:
            st.empty()
    st.markdown(
        "<div class='action-hint'>Mode guide: Text = type directly. Voice = record answer then transcribe before submit.</div>",
        unsafe_allow_html=True,
    )

    # Subsection: CV upload and extracted text editing
    if st.session_state.iv_clear_cv_state:
        st.session_state.iv_cv_text = ""
        st.session_state.iv_cv_file_sig = ""
        st.session_state.iv_clear_cv_state = False

    uploaded_cv = st.file_uploader(
        "Upload CV",
        type=["pdf"],
        key=f"iv_setup_cv_upload_{st.session_state.iv_cv_uploader_nonce}",
        help="Upload your CV to personalize questions and follow-up prompts.",
    )
    if uploaded_cv is not None:
        try:
            file_sig = f"{uploaded_cv.name}:{uploaded_cv.size}"
            if st.session_state.iv_cv_file_sig != file_sig:
                st.session_state.iv_cv_text = parse_pdf_text(uploaded_cv)
                st.session_state.iv_cv_file_sig = file_sig
            st.success("CV loaded.")
        except Exception as exc:
            st.warning(f"Failed to parse PDF CV: {exc}")

    st.text_area(
        "CV Text (auto-filled from PDF, editable)",
        key="iv_cv_text",
        height=180,
        placeholder="Upload CV, or paste profile summary here...",
        help="Review extracted CV text and edit if needed before starting.",
    )
    cv_text = st.session_state.iv_cv_text

    # Subsection: Setup actions (reset/start)
    hint_col, start_col = st.columns([5, 2])
    with hint_col:
        st.markdown("<div class='action-hint'>Need a clean slate? Use Reset Session.</div>", unsafe_allow_html=True)
        reset_clicked = st.button("Reset Session", type="tertiary")
    with start_col:
        start_clicked = st.button("Start Interview", type="primary", use_container_width=True)

    if reset_clicked:
        reset_interview()
        st.rerun()

    if start_clicked:
        if llm is None:
            st.error("Please save an OpenAI API key first.")
        elif job_role == "Please select":
            st.error("Please select a target job role.")
        elif company_type == "Please select":
            st.error("Please select a company type.")
        elif company_size == "Please select":
            st.error("Please select a company size.")
        elif company_location == "Please select":
            st.error("Please select a company location.")
        elif interview_style == "Please select":
            st.error("Please select an interview style.")
        elif plan_duration == "Please select":
            st.error("Please select a plan duration.")
        elif interview_mode == "Please select":
            st.error("Please select an interview mode.")
        elif not cv_text.strip():
            st.error("Please upload CV (or paste profile text).")
        else:
            company_descriptor = f"{company_size} in {company_location}"
            company_profile = build_company_profile(company_type, company_size, company_location)
            try:
                service = InterviewService(llm)
                with st.spinner("Generating your first interview question..."):
                    q = service.generate_first_question(
                        job_role=job_role,
                        company_type=company_type,
                        company_target=company_descriptor,
                        company_profile=company_profile,
                        cv_text=cv_text.strip(),
                        interview_style=interview_style,
                    )
                st.session_state.iv_started = True
                st.session_state.iv_round = 1
                st.session_state.iv_current_question = q.question
                st.session_state.iv_current_intent = q.intent
                st.session_state.iv_summary = None
                st.session_state.iv_records = []
                st.session_state.iv_setup = {
                    "job_role": job_role,
                    "company_type": company_type,
                    "company_size": company_size,
                    "company_location": company_location,
                    "company_profile": company_profile,
                    "interview_style": interview_style,
                    "plan_duration": plan_duration,
                    "interview_mode": interview_mode,
                    "cv_text": cv_text.strip(),
                    "max_rounds": max_rounds,
                }
                st.session_state.iv_nav_page = "Interview Room"
                st.success("Interview started.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to start interview: {exc}")

    if st.session_state.iv_started or st.session_state.iv_records:
        st.markdown("---")
        # Subsection: Quick navigation back to active interview
        c_hint, c_next = st.columns([5, 2])
        with c_hint:
            st.markdown("<div class='action-hint'>Interview in progress. Continue in Interview Room.</div>", unsafe_allow_html=True)
        with c_next:
            if st.button("Go To Interview Room", use_container_width=True, type="primary"):
                go_to_step(1)

# ============================================
# Page 2: Interview Room
# ============================================
elif current_page == "Interview Room":
    st.markdown(
        f"## Interview Room {help_icon('Answer questions one by one. Submit each response for AI scoring and follow-up question generation.')}",
        unsafe_allow_html=True,
    )
    setup = st.session_state.iv_setup or {}
    if not setup:
        st.info("Complete Interview Setup first.")
    else:
        # Subsection: Resolve active setup context for this session
        active_job_role = setup.get("job_role", "Unknown role")
        active_company_type = setup.get("company_type", "General Corporate")
        active_company_size = setup.get("company_size", "Unknown size")
        active_company_location = setup.get("company_location", "Unknown location")
        active_company_target = f"{active_company_size} in {active_company_location}"
        active_company_profile = setup.get(
            "company_profile",
            build_company_profile(active_company_type, active_company_size, active_company_location),
        )
        active_style = setup.get("interview_style", "Mixed")
        active_plan_duration = setup.get("plan_duration", "Within 1 month")
        active_mode = setup.get("interview_mode", "Text")
        active_cv = setup.get("cv_text", "")
        active_max_rounds = int(setup.get("max_rounds", 5))

        if st.session_state.iv_started:
            if st.session_state.iv_current_question:
                # Subsection: Active question prompt
                st.markdown(
                    f"### Question {st.session_state.iv_round} {help_icon('Read the prompt carefully, answer with clear structure, measurable impact, and concise STAR logic.')}",
                    unsafe_allow_html=True,
                )
                if active_mode != "Voice":
                    st.info(st.session_state.iv_current_question)
                    st.caption(f"Intent: {st.session_state.iv_current_intent}")
                st.caption(
                    f"Role: {active_job_role} | Company Type: {active_company_type} | "
                    f"Size: {active_company_size} | Location: {active_company_location} | "
                    f"Style: {active_style} | Mode: {active_mode} | Plan Duration: {active_plan_duration}"
                )
                round_id = st.session_state.iv_round
                session_nonce = st.session_state.get("iv_session_nonce", 0)
                if active_mode == "Voice":
                    # Subsection: Voice-mode question playback + recording/transcription
                    st.markdown(
                        f"#### Question Audio {help_icon('Click once to read the current question aloud immediately.')}",
                        unsafe_allow_html=True,
                    )
                    question_text = st.session_state.iv_current_question.strip()
                    cache_key = f"{session_nonce}|{round_id}|{question_text}"
                    cached_entry = st.session_state.iv_question_audio_cache.get(cache_key, {})
                    cloud_audio_src = cached_entry.get("src", "")

                    if not cloud_audio_src and llm is not None and question_text:
                        try:
                            with st.spinner("Preparing natural voice..."):
                                question_audio, question_mime = synthesize_question_audio(
                                    question_text,
                                    llm,
                                    voice="shimmer",
                                    speed=0.98,
                                )
                            cloud_audio_src = (
                                f"data:{question_mime};base64,{base64.b64encode(question_audio).decode('utf-8')}"
                            )
                            st.session_state.iv_question_audio_cache[cache_key] = {
                                "src": cloud_audio_src,
                                "mime": question_mime,
                                "voice": "shimmer",
                            }
                        except Exception:
                            cloud_audio_src = ""
                            st.caption("Natural cloud voice is temporarily unavailable. Local voice will be used.")

                    render_instant_read_button(
                        st.session_state.iv_current_question,
                        round_id,
                        cloud_audio_src=cloud_audio_src,
                    )

                    with st.expander("Show question", expanded=False):
                        st.info(st.session_state.iv_current_question)

                    st.markdown(
                        f"#### Voice Capture {help_icon('Record your answer, transcribe it to text, then edit before submitting.')}",
                        unsafe_allow_html=True,
                    )
                    audio_key = f"voice_answer_audio_{session_nonce}_{round_id}"
                    transcript_key = f"voice_answer_text_{session_nonce}_{round_id}"
                    if transcript_key not in st.session_state:
                        st.session_state[transcript_key] = ""

                    if hasattr(st, "audio_input"):
                        audio_input = st.audio_input(
                            "Record Your Answer",
                            key=audio_key,
                            help="Record directly in browser, then click Transcribe Voice.",
                        )
                    else:  # pragma: no cover - compatibility fallback
                        audio_input = st.file_uploader(
                            "Upload Voice Answer",
                            type=["wav", "mp3", "m4a", "webm"],
                            key=audio_key,
                            help="Upload an audio file, then click Transcribe Voice.",
                        )

                    t_left, t_mid, t_right = st.columns([1, 2, 1])
                    with t_mid:
                        transcribe_now = st.button(
                            "Transcribe Voice",
                            type="secondary",
                            use_container_width=True,
                            help="Convert recorded audio into editable answer text.",
                        )
                    if transcribe_now:
                        try:
                            transcript = transcribe_audio_answer(audio_input, llm)
                            st.session_state[transcript_key] = transcript
                            st.success("Voice transcribed. Review text before submitting.")
                        except Exception as exc:
                            st.error(str(exc))

                    st.markdown("**Transcribed Answer (editable)**")
                    answer = st.text_area(
                        "Transcribed Answer (editable)",
                        key=transcript_key,
                        height=180,
                        placeholder="Your transcribed answer will appear here...",
                        label_visibility="collapsed",
                        help="Edit the transcript before submission if needed.",
                    )
                    c_left, c_right = st.columns([6, 1])
                    with c_right:
                        clear_transcript = st.button(
                            "Clear Transcript",
                            type="tertiary",
                            key=f"clear_transcript_{session_nonce}_{round_id}",
                            help="Clear current transcript text.",
                        )
                    if clear_transcript:
                        st.session_state[transcript_key] = ""
                        st.rerun()
                else:
                    # Subsection: Text-mode answer input
                    answer = st.text_area(
                        "Your Answer",
                        key=f"answer_round_{session_nonce}_{round_id}",
                        height=180,
                        placeholder="Type your interview answer...",
                        help="Provide your full interview response here. Submit to receive score and follow-up.",
                    )

                # Subsection: Per-question actions
                a1, a2 = st.columns([2, 2])
                with a1:
                    st.markdown(
                        "<div class='action-hint'>Ending now will close the interview and open Performance Report.</div>",
                        unsafe_allow_html=True,
                    )
                    end_now = st.button("End Interview Now", type="tertiary", help="Finish now and generate final report.")
                with a2:
                    submit_answer = st.button("Submit Answer", type="primary", use_container_width=True)

                if submit_answer:
                    if not answer.strip():
                        st.error("Please enter your answer before submitting.")
                    elif llm is None:
                        st.error("Please save an OpenAI API key first.")
                    else:
                        try:
                            # Evaluate current answer and prepare next question if applicable.
                            service = InterviewService(llm)
                            with st.spinner("Evaluating answer and generating follow-up..."):
                                evaluation = service.evaluate_answer(
                                    job_role=active_job_role,
                                    company_type=active_company_type,
                                    company_target=active_company_target,
                                    company_profile=active_company_profile,
                                    cv_text=active_cv,
                                    interview_style=active_style,
                                    question=st.session_state.iv_current_question,
                                    answer=answer.strip(),
                                    round_index=st.session_state.iv_round,
                                )

                            record = {
                                "round": st.session_state.iv_round,
                                "question": st.session_state.iv_current_question,
                                "answer": answer.strip(),
                                "overall_score": evaluation.overall_score,
                                "structure_score": evaluation.structure_score,
                                "clarity_score": evaluation.clarity_score,
                                "impact_score": evaluation.impact_score,
                                "star_score": evaluation.star_score,
                                "strengths": evaluation.strengths,
                                "weaknesses": evaluation.weaknesses,
                                "improved_answer": evaluation.improved_answer,
                                "coach_tip": evaluation.coach_tip,
                                "follow_up_question": evaluation.follow_up_question,
                            }
                            st.session_state.iv_records.append(record)

                            if len(st.session_state.iv_records) >= active_max_rounds:
                                st.session_state.iv_current_question = ""
                                st.session_state.iv_current_intent = ""
                                st.success(
                                    "Answer evaluated. You reached the selected number of questions. "
                                    "Continue to Response Evaluation or end interview for Performance Report."
                                )
                            else:
                                st.session_state.iv_round += 1
                                st.session_state.iv_current_question = evaluation.follow_up_question
                                st.session_state.iv_current_intent = "Follow-up based on your previous answer"
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Evaluation failed: {exc}")

                if end_now and st.session_state.iv_records:
                    if generate_final_summary(setup, llm):
                        go_to_step(3)
            else:
                # Subsection: Interview finished for selected round count
                st.markdown("<div class='action-hint'>Question ended. You can finalize now or continue to Response Evaluation.</div>", unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    st.markdown(
                        "<div class='action-hint'>Ending now will generate report and open Performance Report.</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("End Interview Now", type="tertiary", help="Finish and open Performance Report."):
                        if generate_final_summary(setup, llm):
                            go_to_step(3)
                with b2:
                    if st.button("Go To Response Evaluation", use_container_width=True, type="primary"):
                        go_to_step(2)
        else:
            # Subsection: Not running, but allow navigation to completed outputs
            st.info("Interview is not currently running. Start from Interview Setup or review completed results.")
            if st.session_state.iv_records:
                n_spacer, n_actions = st.columns([5, 2])
                with n_actions:
                    if st.button("Go To Response Evaluation", use_container_width=True, type="primary"):
                        go_to_step(2)
                    if st.button("Go To Performance Report", use_container_width=True, type="primary"):
                        go_to_step(3)

# ============================================
# Page 3: Response Evaluation
# ============================================
elif current_page == "Response Evaluation":
    st.markdown(
        f"## Response Evaluation {help_icon('Review scores and qualitative feedback for each submitted answer. Use this page to compare responses before final report.')}",
        unsafe_allow_html=True,
    )
    if not st.session_state.iv_records:
        st.info("No evaluated responses yet. Complete at least one answer in Interview Room.")
    else:
        # Subsection: Session-level summary metrics
        records = st.session_state.iv_records
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Responses Evaluated", len(records))
        mc2.metric("Average Score", f"{round(sum(r['overall_score'] for r in records) / len(records), 1)}/10")
        mc3.metric("Latest Score", f"{records[-1]['overall_score']}/10")

        # Subsection: Numeric score table (per question)
        st.markdown(
            f"### Score Summary {help_icon('Numeric rubric scores for each answered question. Higher score indicates better performance on that dimension.')}",
            unsafe_allow_html=True,
        )
        history_df = pd.DataFrame(
            [
                {
                    "Q#": int(r["round"]),
                    "Overall": int(r["overall_score"]),
                    "Structure": int(r["structure_score"]),
                    "Clarity": int(r["clarity_score"]),
                    "Impact": int(r["impact_score"]),
                    "STAR": int(r["star_score"]),
                }
                for r in records
            ]
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # Subsection: Expandable qualitative review per question
        st.markdown(
            f"### Per-Question Review {help_icon('Expand each question to inspect your answer and detailed AI evaluation.')}",
            unsafe_allow_html=True,
        )
        for i, rec in enumerate(records, start=1):
            st.markdown(f"#### Question {i}")
            st.caption(rec["question"])
            with st.expander("Your Answer", expanded=False):
                st.write(rec["answer"])
            with st.expander(f"Evaluation Score ({rec['overall_score']}/10)", expanded=False):
                render_evaluation_details(rec)
            st.markdown("---")

        # Subsection: Page navigation actions
        nav1, nav2 = st.columns(2)
        with nav1:
            if st.session_state.iv_started and st.session_state.iv_current_question:
                if st.button("Back To Interview Room", use_container_width=True):
                    go_to_step(1)
        with nav2:
            if st.button("Go To Performance Report", use_container_width=True, type="primary"):
                go_to_step(3)

# ============================================
# Page 4: Performance Report
# ============================================
elif current_page == "Performance Report":
    st.markdown(
        f"## Performance Report {help_icon('Final analytics and improvement guidance across the full interview session.')}",
        unsafe_allow_html=True,
    )
    if not st.session_state.iv_records:
        st.info("No interview records yet. Complete Interview Setup and run at least one question.")
    else:
        # Subsection: Trend chart + compact history table
        setup = st.session_state.iv_setup or {}
        st.markdown(
            f"### Performance Snapshot {help_icon('Trend line of overall score across interview rounds.')}",
            unsafe_allow_html=True,
        )
        render_score_trend(st.session_state.iv_records)

        st.markdown(
            f"### Interview History Summary {help_icon('Compact table of question-by-question scores and short answer summaries.')}",
            unsafe_allow_html=True,
        )
        history_df = pd.DataFrame(
            [
                {
                    "Q#": int(r["round"]),
                    "Overall": int(r["overall_score"]),
                    "Structure": int(r["structure_score"]),
                    "Clarity": int(r["clarity_score"]),
                    "Impact": int(r["impact_score"]),
                    "STAR": int(r["star_score"]),
                    "Question Summary": (r["question"][:78] + "...") if len(r["question"]) > 78 else r["question"],
                    "Answer Summary": (r["answer"][:96] + "...") if len(r["answer"]) > 96 else r["answer"],
                }
                for r in st.session_state.iv_records
            ]
        )
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Q#": st.column_config.NumberColumn("Q#", width="small"),
                "Overall": st.column_config.NumberColumn("Overall", width="small"),
                "Structure": st.column_config.NumberColumn("Structure", width="small"),
                "Clarity": st.column_config.NumberColumn("Clarity", width="small"),
                "Impact": st.column_config.NumberColumn("Impact", width="small"),
                "STAR": st.column_config.NumberColumn("STAR", width="small"),
                "Question Summary": st.column_config.TextColumn("Question Summary", width="large"),
                "Answer Summary": st.column_config.TextColumn("Answer Summary", width="large"),
            },
        )

        if st.session_state.iv_summary:
            # Subsection: Final synthesized report details
            summary = st.session_state.iv_summary
            improvement_items = summary.get("overall_improvement_suggestions") or summary.get("suggested_answers", [])
            st.markdown("---")
            # Subsection: Preview and export actions
            st.markdown(
                f"### Interview Performance Report {help_icon('Structured debrief with strengths, gaps, improvements, and a timeline-based training plan.')}",
                unsafe_allow_html=True,
            )
            st.metric("Overall Score", f"{summary['overall_score']}/10")

            st.markdown("**Strengths**")
            for item in summary.get("strengths", []):
                st.write(f"- {item}")
            st.markdown("**Weaknesses**")
            for item in summary.get("weaknesses", []):
                st.write(f"- {item}")
            st.markdown("**Overall Improvement Suggestions**")
            for item in improvement_items:
                st.write(f"- {item}")
            st.markdown("**Personalized Training Plan**")
            st.caption(f"Timeline selected in setup: {setup.get('plan_duration', 'N/A')}")
            for item in summary.get("personalized_training_plan", []):
                st.write(f"- {item}")

            st.markdown("---")
            st.markdown(
                f"### Report Preview {help_icon('Preview the final report content before downloading the PDF version.')}",
                unsafe_allow_html=True,
            )
            preview_text = build_report_preview(setup, st.session_state.iv_records, summary)
            st.text_area("Preview", value=preview_text, height=300, disabled=True, label_visibility="collapsed")
            report_pdf = build_report_pdf(setup, st.session_state.iv_records, summary)
            st.download_button(
                "Download Report (PDF)",
                data=report_pdf,
                file_name="interview_performance_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            # Subsection: Trigger final summary generation
            st.info("Generate a final report preview, then decide whether to download.")
            if st.button("End Interview Now (Generate Report Preview)", use_container_width=True, type="primary"):
                if generate_final_summary(setup, llm):
                    st.rerun()

        st.markdown("---")
        # Subsection: End-of-flow navigation/reset actions
        r1, r2 = st.columns(2)
        with r1:
            if st.button("Back To Response Evaluation", use_container_width=True):
                go_to_step(2)
        with r2:
            if st.button("Start New Session", use_container_width=True):
                reset_interview()
                go_to_step(0)
