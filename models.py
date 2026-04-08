"""
models.py — Pydantic v2 typed models for the Support Ticket Resolver OpenEnv.

All public types are importable from the project root via:
    from models import Ticket, Observation, Action, Reward, ...
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single message in a support conversation thread."""
    role: Literal["customer", "agent", "system"]
    content: str


class KBArticle(BaseModel):
    """A knowledge-base search result snippet."""
    article_id: str
    title: str
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)


class Ticket(BaseModel):
    """The core support ticket handed to the agent each episode."""
    ticket_id: str
    subject: str
    description: str
    # Ground-truth labels (hidden from agent prompt, used by grader)
    _gt_category: str = ""
    _gt_priority: str = ""
    _gt_action: str = ""   # "resolve" | "escalate" | "credit"


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the agent can see at each step."""
    ticket_id: str
    subject: str
    description: str
    category: Optional[str] = None
    priority: Optional[str] = None
    status: str = "open"                        # open | pending | resolved | escalated
    conversation_history: List[Message] = []
    internal_notes: List[str] = []
    kb_results: List[KBArticle] = []
    available_actions: List[str] = Field(default_factory=list)
    step_count: int = 0
    last_action_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Action — discriminated union
# ---------------------------------------------------------------------------

class ClassifyTicket(BaseModel):
    action_type: Literal["classify_ticket"] = "classify_ticket"
    category: str   # billing | refund | technical | account | shipping | general
    priority: str   # low | normal | high | urgent


class SearchKB(BaseModel):
    action_type: Literal["search_kb"] = "search_kb"
    query: str


class AskClarification(BaseModel):
    action_type: Literal["ask_clarification"] = "ask_clarification"
    question: str


class DraftResponse(BaseModel):
    action_type: Literal["draft_response"] = "draft_response"
    content: str


class Resolve(BaseModel):
    action_type: Literal["resolve"] = "resolve"
    status: Literal["resolved", "closed"] = "resolved"
    final_notes: str = ""


class Escalate(BaseModel):
    action_type: Literal["escalate"] = "escalate"
    department: str   # billing_team | senior_support | fraud_team | technical_team
    reason: str


class AddInternalNote(BaseModel):
    action_type: Literal["add_internal_note"] = "add_internal_note"
    note: str


# The discriminated union — parse with Action.model_validate({"action_type": ..., ...})
Action = Annotated[
    Union[
        ClassifyTicket,
        SearchKB,
        AskClarification,
        DraftResponse,
        Resolve,
        Escalate,
        AddInternalNote,
    ],
    Field(discriminator="action_type"),
]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Step-level reward signal with an explanation string."""
    value: float = Field(ge=-1.0, le=1.0)
    reason: str


# ---------------------------------------------------------------------------
# Episode result (returned by env.close() / end-of-episode)
# ---------------------------------------------------------------------------

class EpisodeResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    success: bool
    steps: int
    rewards: List[float]
    breakdown: dict = Field(default_factory=dict)