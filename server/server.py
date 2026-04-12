"""
server.py — FastAPI server exposing the SupportEnv as an HTTP API.

Endpoints required by the OpenEnv pre-submission validator:
  POST /reset   → Observation JSON
  POST /step    → {observation, reward, done, info}
  GET  /state   → Observation JSON
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from envs.support_env import SupportEnv
from models import (
    Action, AddInternalNote, AskClarification, ClassifyTicket,
    DraftResponse, Escalate, Observation, Resolve, SearchKB,
)

app = FastAPI(
    title="Support Ticket Resolver — OpenEnv",
    version="0.2.0",
    description="Customer support triage environment for LLM agent benchmarking.",
)

# Single shared environment instance (stateful per-session)
_env = SupportEnv()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: str = "easy"


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "env": "support-ticket-resolver-openenv", "version": "0.2.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment and return the initial observation."""
    difficulty = req.difficulty if req.difficulty in ("easy", "medium", "hard") else "easy"
    obs = _env.reset(difficulty)
    return JSONResponse(content=obs.model_dump())


@app.post("/step")
def step(req: StepRequest):
    """Take one action in the environment."""
    try:
        action = _parse_action(req.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

    obs, reward, done, info = _env.step(action)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    })


@app.get("/state")
def state():
    """Return the current observation without advancing the episode."""
    obs = _env.state()
    return JSONResponse(content=obs.model_dump())


@app.get("/grade")
def grade():
    """Return the current episode score."""
    return {"score": _env.grade()}


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

ACTION_MODELS = {
    "classify_ticket": ClassifyTicket,
    "search_kb": SearchKB,
    "ask_clarification": AskClarification,
    "draft_response": DraftResponse,
    "resolve": Resolve,
    "escalate": Escalate,
    "add_internal_note": AddInternalNote,
}


def _parse_action(raw: Dict[str, Any]):
    atype = raw.get("action_type")
    if atype not in ACTION_MODELS:
        raise ValueError(f"Unknown action_type: {atype!r}")
    return ACTION_MODELS[atype].model_validate(raw)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)