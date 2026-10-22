import json
import re
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from rag_engine import hybrid_search


def get_llm(state: "TriageState"):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=state.get("api_key"),
    )


#State
class TriageState(TypedDict, total=False):

    query: str
    api_key: str
    intent: str

    symptoms: list[str]
    duration: str
    severity: str

    urgency: str
    emergency_detected: bool
    emergency_triggers: list[str]
    emergency_negated: list[str]
    recommended_action: str

    retrieved_context: list[str]

    appointment_type: str
    appointment_slot: str

    response: str
    sources: list[str]


#Emergency Keywords
EMERGENCY_WORDS = ["chest pain", "chest tight", "chest tightness", "chest pressure", 
                   "can't breathe", "cant breathe", "difficulty breathing", "trouble breathing", "shortness of breath", "cannot breathe", "unable to breathe",
                    "unconscious", "passed out", "seizure", "heavy bleeding", "bleeding won't stop",
                    "suicide", "suicidal", "kill myself", "want to die", "end my life", "ending my life", "don't want to live", "dont want to live",
                    "face drooping", "face is drooping", "facial droop", "one side of the face droops", "one side of my face is drooping", "drooping on one side",
                    "difficulty speaking", "trouble speaking", "cannot speak clearly", "suddenly unable to speak", 
                    "slurred speech", "speech is slurred", "slurring her words", "slurring his words", "slurring my words",
                    "arm is numb", "left arm numbness", "left arm is numb", "left arm feels numb", "left arm feels numbness",
                    "left arm numb", "arm numbness", "arm feels numb", "left arm weakness"]


#Emergency Safety Check
NEGATION_EXEMPT_WORDS = {
    "suicide", "suicidal", "kill myself", "want to die", "end my life",
    "ending my life", "don't want to live", "dont want to live",
}


NEGATION_PATTERN = re.compile(
    r"\b(?:"
    r"no|not|never|without|don't|dont|do not|"
    r"doesn't|doesnt|does not|didn't|didnt|did not|"
    r"isn't|isnt|is not|aren't|arent|are not|"
    r"wasn't|wasnt|was not|weren't|werent|were not|"
    r"deny|denies|denied|have no|has no|had no"
    r")\b"
    r"(?:[\s,]+[\w'-]+){0,8}[\s,]*$",
    re.IGNORECASE,
)


def _is_negated_mention(text: str, start: int, end: int) -> bool:
    """Detect local symptom denial without carrying negation across contrast."""
    preceding = text[max(0, start - 55):start]
    matched_phrase = text[start:end].lower().strip()

    if matched_phrase in NEGATION_EXEMPT_WORDS:
        return False

    # A contrast usually means the following symptom has a separate meaning.
    contrast_parts = re.split(
        r"\b(?:but|however|although|whereas|yet|except)\b",
        preceding,
        flags=re.IGNORECASE,
    )
    local_preceding = contrast_parts[-1]

    # Uncertainty is not a clear denial.
    if re.search(
        r"\b(?:not sure|unsure|uncertain|can't tell|cannot tell|wonder if)\b",
        local_preceding,
        re.IGNORECASE,
    ):
        return False

    return bool(NEGATION_PATTERN.search(local_preceding))

def check_emergency(state: TriageState):
    """Flag unnegated emergency phrases and retain negated mentions for UI context."""
    text = (state.get("query") or "").lower().replace("’", "'")
    positive_matches = []
    negated_matches = []

    # Longest phrases first makes matching and the resulting display clearer.
    for word in sorted(EMERGENCY_WORDS, key=len, reverse=True):
        phrase = word.lower().replace("’", "'")
        phrase_pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")
        for match in phrase_pattern.finditer(text):
            target = negated_matches if _is_negated_mention(text, match.start(), match.end()) else positive_matches
            if not any(
                phrase in existing
                for existing in target):
                # Remove shorter matches covered by this longer phrase.
                target[:] = [
                    existing for existing in target
                    if existing not in phrase]
                target.append(phrase)

    detected = bool(positive_matches)
    return {
        "urgency": "emergency" if detected else "",
        "emergency_detected": detected,
        "emergency_triggers": positive_matches,
        "emergency_negated": negated_matches,
    }


#Route After Emergency Check
def route_after_check(state: TriageState):
    if state["urgency"] == "emergency":
        return "Escalation"
    return "Router"

#Router Agent
def router_agent(state: TriageState):
    prompt = f"""
    You are a router for a healthcare assistant.

    Patient message:
    {state["query"]}

    Classify the user's request into ONE category only:

    symptom_check
    self_care_question
    booking_request
    out_of_scope

    Definitions:

    symptom_check:
    The user describes symptoms or asks whether they need
    medical care.

    self_care_question:
    The user asks for general wellness or home-care information.

    booking_request:
    The user wants to book, schedule, or see a doctor.

    out_of_scope:
    The request asks for a diagnosis, a prescription, or medicines,
    or is unrelated to healthcare. This applies even if the message
    also describes symptoms -- a request asking "what disease/condition
    is this" or "is this [condition]" is out_of_scope, not symptom_check,
    regardless of what symptoms are mentioned alongside it.

    Reply with ONLY the category name.
    """

    response = get_llm(state).invoke(prompt)
    intent = response.content.strip().lower()
    valid_intents = ["symptom_check","self_care_question","booking_request","out_of_scope"]

    if intent not in valid_intents:
        intent = "symptom_check"
    return {"intent": intent}


#Route After Router
def route_after_router(state: TriageState):
    intent = state["intent"]
    if intent == "symptom_check":
        return "Symptom Extraction"
    elif intent == "self_care_question":
        return "Knowledge"
    elif intent == "booking_request":
        return "Scheduling"
    else:
        return "Out of scope"


#Symptom Extraction Agent
def symptom_extraction_agent(state: TriageState):
    prompt = f"""
    You are an information extraction assistant for a healthcare
    triage system.

    Patient message:
    {state["query"]}

    Extract ONLY information explicitly mentioned by the patient.

    Return ONLY valid JSON in this exact format:

    {{
        "symptoms": [],
        "duration": "",
        "severity": ""
    }}

    Rules:
    - symptoms must be a list of concrete symptoms explicitly reported as present.
    - Do not list a symptom that the patient explicitly denies having.
    - Do not treat vague phrases such as "a recurring health symptom",
      "something feels wrong", or "I'm not sure what to do" as a symptom name.
      If no concrete symptom is named, return an empty symptoms list.
    - duration should contain the duration if mentioned.
    - If duration is not mentioned, use "not specified".
    - severity must be one of:
      mild, moderate, severe, not specified
    - Do not diagnose.
    - Do not invent information.
    """

    response = get_llm(state).invoke(prompt)

    try:
        content = response.content.strip()
        # Handle occasional markdown JSON fences
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        data = json.loads(content)
        symptoms = data.get(
            "symptoms",
            []
        )

        duration = data.get("duration","not specified")
        severity = data.get("severity","not specified")

        if not isinstance(symptoms, list):
            symptoms = []
        valid_severities = ["mild","moderate","severe","not specified"]
        if severity not in valid_severities:
            severity = "not specified"

        symptoms = [str(item).strip() for item in symptoms if str(item).strip()]
        # Guard against an extractor repeating an emergency term that the
        # deterministic safety check classified as explicitly negated.
        negated_terms = [str(term).lower() for term in state.get("emergency_negated", [])]
        symptoms = [
            symptom for symptom in symptoms
            if not any(
                term == symptom.lower()
                or term in symptom.lower()
                or symptom.lower() in term
                for term in negated_terms
            )
        ]
        return {
            "symptoms": symptoms,
            "duration": duration,
            "severity": severity,
        }

    except Exception:
        # If extraction fails, continue through the existing triage path.
        return {
            "symptoms": [],
            "duration": "not specified",
            "severity": "not specified",
        }


#Triage Agent
def triage_agent(state: TriageState):
    prompt = f"""
    You are a healthcare triage assistant.

    Your job is NOT to diagnose the patient.

    Patient message:
    {state["query"]}

    Symptoms:
    {state.get("symptoms", [])}

    Duration:
    {state.get("duration", "not specified")}

    Severity:
    {state.get("severity", "not specified")}

    Classify urgency into ONE category:

    emergency
    urgent
    routine
    self_care

    Definitions:

    emergency:
    Potentially life-threatening and requires immediate
    emergency care.

    urgent:
    Should be evaluated by a healthcare professional soon,
    generally within 24 hours.

    routine:
    Should be evaluated by a healthcare professional within
    about a week, but does not appear immediately urgent.
    Choose this when symptoms are persistent or recurring, or
    when the symptom itself could reflect something more
    serious and the patient has not given enough detail to be
    confident that self-care alone is sufficient (for example
    unexplained pain, a lump, or a vague symptom like
    "stomach pain" or "something feels wrong").

    self_care:
    Appears suitable for general self-care guidance. Choose
    this for common, usually mild symptoms mentioned on their
    own with no worrying detail attached -- for example an
    ordinary headache, a sore throat, a runny nose, a mild
    cough, or minor fatigue. These are self_care by default
    even without extra detail, since they are mild and
    short-lived for most people. Only move these to a higher
    tier if the patient's own words add a red-flag detail
    (e.g. severe, worsening, lasting more than a week).

    Rules:

    - Do not diagnose.
    - Do not recommend prescription medicines.
    - If you are uncertain between two urgency levels,
      choose the HIGHER urgency level rather than the lower one.

    Reply with ONLY one category.
    """

    response = get_llm(state).invoke(prompt)
    urgency = response.content.strip().lower()

    valid_urgencies = ["emergency","urgent","routine","self_care"]
    if urgency not in valid_urgencies:
        urgency = "urgent"

    if (
        urgency == "emergency"
        and state.get("emergency_negated")
        and not state.get("emergency_triggers")
        and not state.get("symptoms")
    ):
        urgency = "self_care"

    return {"urgency": urgency}

#Route After Triage
def route_after_triage(state: TriageState):
    urgency = state["urgency"]
    if urgency == "emergency":
        return "Escalation"
    elif urgency in ("urgent", "routine"):
        return "Scheduling"
    else:
        return "Knowledge"

#Escalation Agent
def escalation_agent(state: TriageState):
    message = (
        "This may require immediate medical attention. "
        "Please call your local emergency number or go to the "
        "nearest emergency department immediately. "
        "Do not rely on this assistant for emergency care."
    )

    return {"recommended_action":"Seek emergency medical care immediately.","response":message}

#Knowledge / RAG Agent
def knowledge_agent(state: TriageState):
    query = state["query"]
    # Retrieve relevant evidence using Hybrid RAG
    results = hybrid_search(
    query=query,
    top_k=3,
    api_key=state.get("api_key"),
)

    # Build context for LLM
    context_parts = []
    sources = []
    for result in results:
        context_parts.append(
            f"Title: {result['title']}\n"
            f"Topic: {result['topic']}\n"
            f"Information: {result['text']}\n"
            f"Source: {result['source']}\n"
            f"Source URL: {result['source_url']}"
        )
        sources.append(result["source"])
    context = "\n\n".join(context_parts)

    # Generate answer using retrieved evidence
    prompt = f"""
    You are a healthcare information assistant.
    Use ONLY the provided retrieved evidence.
    Retrieved evidence:

    {context}

    User question:

    {query}

    Give a short and simple general self-care response.

    Rules:
    - Do not diagnose.
    - Do not prescribe medicines.
    - Do not invent information.
    - Do not provide information that is not supported
      by the retrieved evidence.
    - If the retrieved evidence is insufficient,
      clearly say that the available information is limited.
    - Encourage professional medical care when the
      retrieved evidence indicates that it is needed.
    """

    response = get_llm(state).invoke(prompt)

    return {

        "retrieved_context": [
            result["text"]
            for result in results
        ],

        "sources": list(
            dict.fromkeys(sources)
        ),

        "response": response.content
    }

#Scheduling Agent
with open("slots.json", "r", encoding="utf-8") as f:
    SLOTS = json.load(f)

def scheduling_agent(state: TriageState):
    urgency = state.get("urgency")
    if urgency == "urgent":
        slot = SLOTS["urgent"]
        appointment_type = ("Urgent consultation")

    elif urgency == "routine":
        slot = SLOTS["routine"]
        appointment_type = ("Routine consultation")

    else:
        # Used when user directly asks for booking
        slot = SLOTS["routine"]
        appointment_type = ("General consultation")

    message = (
        f"Suggested appointment:\n"
        f"Doctor: {slot['doctor']}\n"
        f"Specialty: {slot['specialty']}\n"
        f"Time: {slot['time']}\n\n"

        f"This is a simulated appointment slot and does not "
        f"represent a real booking."
    )

    return {
        "appointment_type":
            appointment_type,
        "appointment_slot":
            slot["time"],
        "response":
            message
    }


#Out of Scope Agent
def outofscope_agent(state: TriageState):
    message = (
        "I can't provide a diagnosis or prescribe medicines. "
        "I can help with basic symptom urgency, general "
        "self-care information, or a simulated appointment slot."
    )
    return {"response": message}


#Build LangGraph
graph = StateGraph(TriageState)

graph.add_node("Check Emergency",check_emergency)
graph.add_node("Router",router_agent)
graph.add_node("Symptom Extraction",symptom_extraction_agent)
graph.add_node("Triage",triage_agent)
graph.add_node("Escalation",escalation_agent)
graph.add_node("Scheduling",scheduling_agent)
graph.add_node("Knowledge",knowledge_agent)
graph.add_node("Out of scope",outofscope_agent)

#Graph Edges
# START → Emergency Check
graph.add_edge(START,"Check Emergency")

# Emergency Check → Escalation / Router
graph.add_conditional_edges("Check Emergency",route_after_check)

# Router → Symptom Extraction / Knowledge / Scheduling / Out of scope
graph.add_conditional_edges("Router",route_after_router)

# Symptom Extraction → Triage
graph.add_edge("Symptom Extraction","Triage")

# Triage → Escalation / Scheduling / Knowledge
graph.add_conditional_edges("Triage",route_after_triage)


#Terminal Nodes
graph.add_edge("Escalation",END)
graph.add_edge("Scheduling",END)
graph.add_edge("Knowledge",END)
graph.add_edge("Out of scope",END)


#Compile Graph
app = graph.compile()
