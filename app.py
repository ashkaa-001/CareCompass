import os
import time
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # Keeps relative paths used by the existing agent compatible.
st.set_page_config(page_title="CareCompass", page_icon="🧭", layout="wide")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from agent import app as triage_graph
except Exception as e:
    st.error("CareCompass couldn't load its workflow.")
    st.exception(e)
    st.stop()

DISCLAIMER = (
    "CareCompass provides general health information, not a diagnosis or a substitute for a "
    "qualified clinician. It cannot reliably rule out emergencies. If you think someone may "
    "be in immediate danger, contact your local emergency service."
)

URGENCY = {
    "emergency": ("🔴 Emergency", "red", "Emergency", "Get emergency help now. Do not wait for an online response."),
    "urgent": ("🟠 Urgent", "orange", "Prompt medical assessment", "Consider contacting a healthcare professional promptly."),
    "routine": ("🔵 Routine", "blue", "Arrange routine care", "Consider arranging a non-emergency appointment if symptoms persist or concern you."),
    "self_care": ("🟢 Self-care", "green", "General self-care information", "Monitor how you feel and seek medical advice if symptoms worsen or concern you."),
}

INTENT = {
    "symptom_check": ("Symptom review", "Review the guidance below and consider the next steps."),
    "self_care_question": ("Self-care information", "General information based on the available guidance."),
    "booking_request": ("Appointment options", "The options below are examples only; no appointment is booked."),
    "out_of_scope": ("Outside this assistant's scope", "CareCompass cannot diagnose conditions or prescribe treatment."),
}

EXAMPLES = {
    # Keep duration out of the example text so the optional duration control is consistent.
    "Chest discomfort": "I have chest discomfort and my left arm feels numb.",
    "Fever": "I have a fever and ear pain.",
    "Stomach pain": "I've had stomach pain on and off.",
    "Sore throat": "I have sore throat and a runny nose. What general self-care may help?",
    "Appointment": "I'd like to see what sample doctor appointment options are available.",
}
DURATIONS = ["Not sure", "Started today", "1-3 days", "About a week", "More than a week", "More than a month"]
SEVERITIES = ["Not sure", "Mild", "Moderate", "Severe"]

st.session_state.setdefault("symptom_text", "")
st.session_state.setdefault("run", None)
st.session_state.setdefault("duration_choice", DURATIONS[0])
st.session_state.setdefault("severity_choice", SEVERITIES[0])
st.session_state.setdefault("api_key_validated", False)
st.session_state.setdefault("api_key_fingerprint", "")
st.session_state.setdefault("api_key_input", os.environ.get("OPENAI_API_KEY", ""))


def choose_example():
    choice = st.session_state.get("example_pick")
    if choice:
        st.session_state.symptom_text = EXAMPLES[choice]


def start_over():
    """Reset the whole symptom form. The API key is deliberately kept for this session."""
    st.session_state.run = None
    st.session_state.symptom_text = ""
    st.session_state.example_pick = None
    st.session_state.duration_choice = DURATIONS[0]
    st.session_state.severity_choice = SEVERITIES[0]


def clear_api_key():
    st.session_state.api_key_input = ""
    st.session_state.api_key_validated = False
    st.session_state.api_key_fingerprint = ""


def parse_slot(text):
    parsed = {}
    for line in (text or "").splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key in {"Doctor", "Specialty", "Time"}:
            parsed[key] = value.strip()
    return parsed


def run_pipeline(query, api_key):
    state = {
        "query": query,
        "api_key": api_key,
        "intent": "",
        "symptoms": [],
        "duration": "not specified",
        "severity": "not specified",
        "urgency": "",
        "emergency_detected": False,
        "emergency_triggers": [],
        "emergency_negated": [],
        "recommended_action": "",
        "retrieved_context": [],
        "appointment_type": "",
        "appointment_slot": "",
        "response": "",
        "sources": [],
    }
    final = dict(state)
    steps = []
    start = last = time.perf_counter()

    for chunk in triage_graph.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            now = time.perf_counter()
            steps.append((str(node), round(now - last, 2)))
            last = now
            final.update(update or {})

    final["api_key"] = ""  # Never retain the API key in the displayed/stored result.
    return {"state": final, "steps": steps, "elapsed": round(time.perf_counter() - start, 2)}


def validate_api_key(candidate):
    """Validate the key without displaying or logging it."""
    from openai import OpenAI
    client = OpenAI(api_key=candidate, timeout=12.0, max_retries=0)
    client.models.list()


# Sidebar: concise setup and privacy information.
with st.sidebar:
    st.markdown("### 🧭 CareCompass")
    st.caption("Healthcare guidance and symptom triage")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Describe your symptoms\n2. Review the suggested next step\n3. Check the guidance and its sources")

    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        key="api_key_input",
        placeholder="Paste API key",
        help="Stays in this session and is kept when you press Start over.",
    ).strip()
    # A changed key must be validated again; don't equate a non-empty field with a valid key.
    key_fingerprint = str(hash(api_key)) if api_key else ""
    if key_fingerprint != st.session_state.api_key_fingerprint:
        st.session_state.api_key_validated = False
        st.session_state.api_key_fingerprint = key_fingerprint

    key_col1, key_col2 = st.columns(2)
    with key_col1:
        if st.button("Validate key", use_container_width=True):
            try:
                validate_api_key(api_key)
                st.session_state.api_key_validated = True
                st.success("API key validated for this session.")
            except Exception:
                st.session_state.api_key_validated = False
                st.error("Couldn't validate this key. Check it and try again.")
    with key_col2:
        st.button("Clear key", use_container_width=True, on_click=clear_api_key)

    if st.session_state.api_key_validated and api_key:
        st.caption("Connection check passed")
    else:
        st.caption("Add a key to run the AI workflow.")

    st.divider()
    st.markdown("**Privacy reminder**")
    st.caption("Avoid entering names, addresses, ID numbers, or other identifying details.")
    st.caption("Built with LangGraph, OpenAI and ChromaDB")

# Header
st.title("🧭 CareCompass")
st.write("A simple way to review symptoms, find general health guidance, and explore sample appointment options.")
st.divider()

# Intake area
st.subheader("Tell us what's going on")
st.write("Include what you feel, when it started, and any changes you've noticed. You can leave the optional details blank.")
with st.container(border=True):
    st.text_area(
        "Symptoms or health question",
        key="symptom_text",
        height=130,
        max_chars=1000,
        placeholder="For example: I've had a headache since yesterday and it gets worse when I stand up.",
    )
    st.pills("Example prompts (optional)", list(EXAMPLES.keys()), key="example_pick", on_change=choose_example)
    with st.expander("Add optional details"):
        detail1, detail2 = st.columns(2)
        duration = detail1.selectbox("How long has this been happening?", DURATIONS, key="duration_choice")
        severity = detail2.selectbox("How severe does it feel?", SEVERITIES, key="severity_choice")
    go_col, clear_col = st.columns([3, 1])
    with go_col:
        go = st.button("Review symptoms", type="primary", use_container_width=True)
    with clear_col:
        st.button("Start over", on_click=start_over, use_container_width=True)

# Run pipeline. API key presence is required for the AI path; validation is encouraged,
# but a key may be supplied by environment configuration and checked on first use.
if go:
    query = st.session_state.symptom_text.strip()
    if not query:
        st.warning("Describe your symptoms or choose an example prompt first.")
    elif not api_key:
        st.warning("Add your OpenAI API key in the sidebar to run the AI workflow.")
    else:
        if duration != "Not sure":
            query += f" Duration: {duration.lower()}."
        if severity != "Not sure":
            query += f" Severity: {severity.lower()}."
        try:
            with st.spinner("Reviewing your message…"):
                st.session_state.run = run_pipeline(query, api_key)
        except Exception:
            st.session_state.run = None
            st.error("The review couldn't be completed. Check your API key and the terminal logs, then try again.")

# Results: put the main guidance on the page rather than hiding it behind a tab.
run = st.session_state.run
if run:
    result = run["state"]
    urgency = (result.get("urgency") or "").strip().lower()
    response = result.get("response", "") or "No response was returned by the workflow."

    
    st.subheader("Your result")

    if urgency == "emergency":
        st.error("**Possible emergency**\n\n" + response)
        triggers = result.get("emergency_triggers") or []
        if triggers:
            st.caption("Emergency-related indicators detected: " + ", ".join(triggers))
        with st.expander("Why this message appeared"):
            if triggers:
                st.write("A safety rule matched an emergency-related phrase in your message. This is a precaution, not a diagnosis.")
            else:
                st.write("The AI's own assessment flagged your symptoms as a possible emergency. This is a precaution, not a diagnosis.")
        st.caption(DISCLAIMER)
        st.stop()

    negated_mentions = result.get("emergency_negated") or []
    if negated_mentions:
        st.info(
            "Your message mentions an emergency-related phrase in a negative context: "
            + ", ".join(negated_mentions)
            + ". The keyword check did not treat that mention alone as a positive emergency indicator. "
            "This does not rule out an emergency; seek urgent help if you have concerning symptoms."
        )

    if urgency in URGENCY:
        badge, color, label, next_step = URGENCY[urgency]
        with st.container(border=True):
            st.markdown(f":{color}-badge[{badge}]")
            st.markdown(f"### {label}")
            st.write(next_step)
    else:
        label, intro = INTENT.get(result.get("intent", ""), ("General result", "Review the information below."))
        with st.container(border=True):
            st.markdown(f"### {label}")
            st.write(intro)

    if result.get("symptoms"):
        st.markdown("**Symptoms identified from your message**")
        st.markdown(" ".join(f":green-badge[{str(item).replace(']', '')}]" for item in result["symptoms"]))

    if result.get("appointment_slot"):
        heading = "Suggested appointment"
    elif result.get("intent") == "self_care_question" or urgency == "self_care":
        heading = "Self-care guidance"
    else:
        heading = "Guidance"
    st.markdown(f"### {heading}")

    slot = parse_slot(response) if result.get("appointment_slot") else {}
    if slot:
        with st.container(border=True):
            if result.get("appointment_type"):
                st.markdown(f"**{result['appointment_type']}**")
            st.markdown(
                f"**Doctor:** {slot.get('Doctor', 'n/a')}  \n"
                f"**Specialty:** {slot.get('Specialty', 'n/a')}  \n"
                f"**Time:** {slot.get('Time', 'n/a')}"
            )
        st.caption("Sample slot only. No real booking is made.")
    else:
        st.markdown(response.replace("\n", "  \n"))

    sources = result.get("sources") or []
    if sources:
        with st.expander("Sources used by the assistant"):
            for source in sources:
                st.markdown(f"- {source}")
    else:
        st.caption("No sources were returned for this response.")

    with st.expander("Developer diagnostics"):
        for index, (step, seconds) in enumerate(run.get("steps", []), 1):
            st.write(f"{index}. {step} — {seconds:.2f}s")
        st.caption(f"Total workflow time: {run.get('elapsed', 0):.2f}s")

    st.caption(DISCLAIMER)