---
title: Support Ticket Resolver OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🎫 Support Ticket Resolver — OpenEnv

> A production-grade customer support triage environment for benchmarking LLM agents on multi-step classification, knowledge-base lookup, policy-aware escalation, and empathetic response drafting.

---

## Overview

Support Ticket Resolver is an **OpenEnv**-compliant environment that simulates a real-world customer support workflow. An LLM agent receives a support ticket and must:

1. **Classify** the ticket (category + priority)
2. **Search** a knowledge base for relevant policy information
3. **Draft** a personalised, policy-compliant response or **escalate** appropriately
4. **Resolve** or **escalate** the ticket with full justification

The environment provides **dense, step-level rewards** to guide learning, and a final **graded score [0, 1]** for evaluation.

---

## Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `easy_triage` | Easy | Single-turn billing/refund ticket → classify + resolve | 6 |
| `medium_kb_reply` | Medium | UPI payment failure → KB search + draft personalised reply | 8 |
| `hard_policy_escalation` | Hard | ₹12,000 fraud claim → policy check + escalate to correct team | 12 |

### Easy — Duplicate Charge Refund
A customer reports a duplicate ₹499 subscription charge. The agent must correctly classify it as `billing / normal` and resolve with a refund confirmation.

### Medium — UPI Payment Not Reflected
A customer paid ₹1,299 via UPI two days ago but the account hasn't upgraded. The agent must search the KB for UPI reversal policy, optionally ask one clarification, draft a personalised reply, and resolve.

### Hard — Fraud Escalation (Policy-Constrained)
An angry customer reports an unauthorized ₹12,000 charge. Per company policy, refunds >₹5,000 require escalation to the `fraud_team`. Resolving without escalating is a **policy violation** (−0.30 penalty).

---

## Observation Space

```python
class Observation(BaseModel):
    ticket_id: str
    subject: str
    description: str
    category: Optional[str]           # set after classify_ticket
    priority: Optional[str]           # set after classify_ticket
    status: str                       # open | pending | resolved | escalated | closed | timeout
    conversation_history: List[Message]
    internal_notes: List[str]
    kb_results: List[KBArticle]       # populated after search_kb
    available_actions: List[str]
    step_count: int
    last_action_error: Optional[str]  # error from previous step, if any
```

---

## Action Space

All actions are Pydantic models with a `action_type` discriminator field:

| Action | Key Fields | Purpose |
|--------|-----------|---------|
| `classify_ticket` | `category`, `priority` | Label the ticket |
| `search_kb` | `query` | Search the hardcoded knowledge base |
| `ask_clarification` | `question` | Ask the customer a follow-up (medium only) |
| `draft_response` | `content` | Write a reply to the customer |
| `resolve` | `status`, `final_notes` | Close the ticket |
| `escalate` | `department`, `reason` | Hand off to a specialist team |
| `add_internal_note` | `note` | Record internal context |

**Valid categories:** `billing`, `refund`, `technical`, `account`, `shipping`, `general`  
**Valid priorities:** `low`, `normal`, `high`, `urgent`  
**Valid departments:** `billing_team`, `senior_support`, `fraud_team`, `technical_team`

---

## Reward Function (Dense)

| Event | Reward |
|-------|--------|
| Correct category | +0.20 |
| Correct priority | +0.15 |
| KB search (any result) | +0.15 |
| Draft response (≥20 chars) | +0.20 |
| Draft response after KB | +0.25 |
| Correct resolve | +0.25 |
| Correct escalate (right dept) | +0.35 |
| Correct escalate (wrong dept) | +0.20 |
| Finish in ≤ half max_steps | +0.05 |
| Step overhead | −0.01 |
| Wrong category | −0.05 |
| Wrong priority | −0.05 |
| Repeated action (≥3 times) | −0.10 |
| Invalid action | −0.20 |
| Policy violation (resolve instead of escalate) | −0.30 |
| Unnecessary escalation | −0.15 |
| >1 clarification (medium) | −0.10 |

---

## Grader Breakdown

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| Correct category | 0.25 | 0.25 | 0.25 |
| Correct priority | 0.20 | 0.20 | 0.20 |
| KB searched | — | 0.15 | 0.10 |
| Response drafted | 0.15 | 0.20 | — |
| Resolved | 0.30 | 0.15 | — |
| Escalated (correct dept) | — | — | 0.40 |
| Internal note | — | — | 0.05 |
| Efficiency bonus | 0.10 | — | — |

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-org/support-ticket-resolver-openenv
cd support-ticket-resolver-openenv

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# 5. Run tests
python -m pytest tests/ -v

# 6. Run inference (all 3 tasks)
python inference.py

# 7. Run a single task
TASK=easy python inference.py
```

---

## Running the Server Locally

```bash
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

The server exposes:
- `POST /reset`  — start a new episode (`{"difficulty": "easy|medium|hard"}`)
- `POST /step`   — take an action (`{"action": {...}}`)
- `GET  /state`  — current observation
- `GET  /grade`  — current score

---

## Docker

```bash
docker build -t support-ticket-resolver .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="hf_xxx" \
  support-ticket-resolver
```

---

## Deploying to Hugging Face Spaces

1. Create a new Docker Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Push this repository as the Space source
3. Add `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` as Space Secrets
4. The Space will build and expose the `/reset`, `/step`, `/state` endpoints

---

## Pre-submission Validation

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space .
```

All three checks must pass:
- ✅ HF Space is live and responds to `/reset`
- ✅ Docker build succeeds
- ✅ `openenv validate` passes

---

## Baseline Scores (Deterministic Agent)

| Task | Steps | Score |
|------|-------|-------|
| easy_triage | 3 | 0.950 |
| medium_kb_reply | 4 | 0.750 |
| hard_policy_escalation | 3 | 0.850 |

---

## Project Structure

```
.
├── inference.py          # Root-level LLM agent inference script
├── server.py             # FastAPI server (OpenEnv HTTP interface)
├── models.py             # Pydantic v2 models (Observation, Action, Reward)
├── openenv.yaml          # OpenEnv metadata and task spec
├── Dockerfile            # Multi-stage production Docker image
├── requirements.txt
├── README.md
├── envs/
│   ├── __init__.py
│   └── support_env.py    # SupportEnv with 3 tasks, reward logic, graders
└── tests/
    └── test_env.py       # Deterministic pytest suite
```

---

## License

MIT