# CareCompass 🧭

**Healthcare Symptom Triage & Appointment Assistant** — an educational capstone project that helps users review reported symptoms, explore general self-care information, and view simulated appointment options.

> **Important:** CareCompass is a non-diagnostic educational demo. It is not a medical device, does not replace a qualified healthcare professional, and cannot reliably rule out an emergency. If you think someone may be in immediate danger, contact your local emergency service.

## Problem Statement

People can be unsure whether their symptoms require emergency attention, a general-practitioner visit, or self-care. CareCompass demonstrates a safety-focused workflow for routing symptom-related requests, providing source-grounded general health information, and showing mock appointment options according to urgency.

## Features

- **Deterministic emergency screening:** checks messages for configured emergency-related phrases before routing the request. When a phrase is detected as a positive emergency indicator, the workflow returns a fixed safety response without calling the LLM on that path.
- **Request routing:** classifies a request as a symptom check, self-care question, appointment request, or out-of-scope request.
- **Symptom extraction and urgency classification:** extracts explicitly reported symptoms and uses the LLM to classify non-emergency symptom requests as emergency, urgent, routine, or self-care.
- **Retrieval-augmented generation (RAG):** combines BM25 keyword retrieval with ChromaDB vector retrieval and reciprocal-rank fusion to find relevant entries in the local health-guideline dataset.
- **Source display:** presents the source names returned with self-care guidance.
- **Simulated scheduling:** displays predefined sample appointment options from `slots.json`; it does not contact a clinic or make a real booking.
- **Streamlit interface:** provides example prompts, optional duration and severity inputs, API-key validation, results, sources, and workflow timing diagnostics.
- **Safety and privacy reminders:** displays a non-diagnostic disclaimer and encourages users not to enter identifying details.

## Architecture

```text
User message
    |
    v
Deterministic emergency phrase check
    | positive emergency indicator
    +------------------------------> Fixed escalation response (no LLM call)
    |
    v
Router Agent (LLM)
    |
    +--> Symptom Extraction --> Triage Agent
    |                            |
    |                            +--> Emergency --> Fixed escalation response
    |                            +--> Urgent / Routine --> Scheduling Agent
    |                            +--> Self-care --> Knowledge Agent (RAG)
    |
    +--> Self-care question ------> Knowledge Agent (RAG)
    +--> Appointment request -----> Scheduling Agent
    +--> Out-of-scope request ----> Scope-limitation response
```

### RAG flow

1. `guidelines.json` provides the health-guideline records.
2. BM25 ranks documents using lexical matching.
3. ChromaDB stores embeddings generated with OpenAI's `text-embedding-3-small` model and ranks semantically similar documents.
4. Reciprocal-rank fusion combines the two rankings.
5. The Knowledge Agent gives the retrieved evidence to `gpt-4o-mini` and asks it to answer using only that evidence. Source names are returned to the UI.

The local ChromaDB index is created on first retrieval and stored in `chroma_db/`. If you change the guideline records after the index has been created, rebuild or clear the local index so it reflects the updated data.

## Tech Stack

- **Python**
- **Streamlit** — user interface
- **LangGraph** — workflow orchestration and conditional routing
- **OpenAI `gpt-4o-mini`** — request routing, extraction, urgency classification, and evidence-based response generation
- **OpenAI `text-embedding-3-small`** — embeddings for semantic retrieval
- **ChromaDB** — persistent local vector store
- **rank-bm25** — lexical retrieval
- **python-dotenv** — optional loading of environment variables from `.env`

## Repository Structure

The application expects these files to be in the same project directory:

```text
CareCompass/
├── app.py              # Streamlit interface
├── agent.py            # LangGraph workflow and agents
├── rag_engine.py       # Hybrid BM25 + ChromaDB retrieval
├── guidelines.json     # Public-health guidance records used by RAG
├── slots.json          # Simulated appointment options
├── requirements.txt    # Python dependencies
├── README.md            # Project documentation
└── chroma_db/           # Local persistent vector index, generated at runtime
```

A Colab notebook or evaluation materials may be stored separately if included in your repository.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashkaa-001/CareCompass.git
cd CareCompass
```

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure your OpenAI API key

You can enter your API key in the app's sidebar. The app provides a **Validate key** button. Alternatively, you can place the key in a local `.env` file:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
```

Keep `.env` private and add it to `.gitignore`. **Never commit an API key to GitHub.** The app uses the key for OpenAI model calls and embeddings. API usage may incur charges according to your OpenAI account and model pricing.

The current app reads `.env` with `python-dotenv` when that package is installed. If you enter the key in the sidebar, it is held in Streamlit session state for the session.

### 5. Run the app

From the project directory:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## How to Use

1. Start the Streamlit app.
2. Enter your OpenAI API key in the sidebar and select **Validate key**.
3. Enter a symptom description or health question, or select an example prompt.
4. Optionally provide symptom duration and severity.
5. Select **Review symptoms**.
6. Review the result, urgency label, general guidance, and any sources or simulated appointment details.

Example prompts are included in the interface for chest discomfort, fever, stomach pain, sore throat, and appointment options. Do not use examples as a substitute for real emergency care.

## Safety, Limitations, and Responsible Use

- **Not diagnostic:** the system does not establish a diagnosis or prescribe treatment.
- **Emergency screening is limited:** the deterministic pre-filter uses a configured phrase list and basic negation handling. It can miss emergencies or flag phrases inappropriately. It is not a substitute for emergency services or clinical assessment.
- **Emergency response is fixed:** when the deterministic check flags an emergency indicator, the workflow returns a hard-coded escalation message and bypasses the LLM for that path.
- **LLM outputs can be wrong:** routing, extraction, and urgency classification are model-generated and may be inaccurate or incomplete.
- **RAG is limited to the available dataset:** responses depend on the content and relevance of records in `guidelines.json` and the local index. Retrieval does not guarantee completeness or correctness.
- **Scheduling is simulated:** the doctors and times in `slots.json` are examples only. No real appointment is booked and no clinic availability is checked.
- **Privacy:** avoid entering names, addresses, contact details, medical record numbers, or other identifying information. Requests sent to OpenAI are subject to the applicable service terms and privacy policies.
- **Disclaimer:** CareCompass provides general health information, not a diagnosis or a substitute for a qualified clinician. It cannot reliably rule out emergencies. If you think someone may be in immediate danger, contact your local emergency service.

This project is an educational demonstration of agentic workflow design and safety patterns, not a real-world healthcare service.

## Evaluation and Testing

The project can be evaluated with a small test set covering:

| Area | Example checks |
|---|---|
| Emergency routing | A positive emergency phrase triggers the fixed escalation response before LLM routing. |
| Negation handling | A clearly denied phrase alone should not be treated as a positive emergency indicator; a separate positive red flag should still be escalated. |
| Intent routing | Symptom checks, self-care questions, appointment requests, and out-of-scope requests follow the expected paths. |
| Urgency classification | Common mild symptoms, persistent symptoms, and severe or concerning descriptions are routed to suitable urgency tiers. |
| RAG grounding | Self-care responses use retrieved guideline records and show source names. |
| Scheduling | Appointment requests show a sample slot and explicitly state that no real booking was made. |
| Safety language | User-facing results include a non-diagnostic disclaimer, and emergency results display the fixed escalation guidance. |

These are suggested evaluation criteria, not a claim that every scenario has been exhaustively validated. The emergency phrase list and LLM-based classification should be tested further before any broader use.

## Future Improvements

- Expand and review the emergency phrase and negation test suite with qualified clinical input.
- Add automated regression tests for routing, retrieval, and safety-critical behavior.
- Add dataset versioning and a clear command or utility to rebuild the ChromaDB index when guideline records change.
- Improve retrieval evaluation and source-level quality checks.
- Replace mock scheduling with a carefully scoped integration only if privacy, security, and clinical workflow requirements are addressed.
- Add deployment guidance, monitoring, and formal safety review before any real-world use.

## Acknowledgements

The guideline dataset includes general public-health information from sources such as **MedlinePlus (NIH/NLM)** and the **U.S. Centers for Disease Control and Prevention (CDC)**. Source URLs are stored with the individual records in `guidelines.json`.

---

