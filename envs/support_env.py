"""
envs/support_env.py — SupportEnv implementing the OpenEnv interface.

Three difficulty levels:
  easy   — single-turn triage + resolve
  medium — KB search required, optional clarification, draft reply
  hard   — multi-turn thread, policy constraints, escalation logic
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action, AddInternalNote, AskClarification, ClassifyTicket,
    DraftResponse, Escalate, KBArticle, Message, Observation,
    Resolve, Reward, SearchKB, Ticket,
)

# ---------------------------------------------------------------------------
# Knowledge Base (hardcoded for determinism)
# ---------------------------------------------------------------------------

KB: List[KBArticle] = [
    KBArticle(
        article_id="KB001",
        title="Refund Policy",
        snippet=(
            "Refunds are processed within 5–7 business days. "
            "Amounts above ₹5,000 require supervisor approval. "
            "Duplicate charges are always eligible for full refund."
        ),
        relevance=0.9,
    ),
    KBArticle(
        article_id="KB002",
        title="UPI Payment Failure",
        snippet=(
            "If a UPI payment fails but money is debited, it auto-reverses within 48 hours. "
            "If not reversed, raise a dispute with your bank referencing the UPI transaction ID."
        ),
        relevance=0.85,
    ),
    KBArticle(
        article_id="KB003",
        title="Account Suspension",
        snippet=(
            "Accounts are suspended after 3 failed login attempts. "
            "Self-service unlock is available via OTP to the registered mobile. "
            "Persistent lockouts should be escalated to the account team."
        ),
        relevance=0.8,
    ),
    KBArticle(
        article_id="KB004",
        title="Subscription Upgrade/Downgrade",
        snippet=(
            "Plan changes take effect from the next billing cycle. "
            "Prorated credits are issued automatically for downgrades."
        ),
        relevance=0.75,
    ),
    KBArticle(
        article_id="KB005",
        title="Technical Troubleshooting — App Crash",
        snippet=(
            "Force-close the app, clear cache, and reopen. "
            "If the issue persists after 2 attempts, collect device logs and escalate to technical_team."
        ),
        relevance=0.7,
    ),
]

# Simple keyword → KB article mapping
KB_INDEX: Dict[str, List[str]] = {
    "refund": ["KB001"],
    "duplicate": ["KB001"],
    "charge": ["KB001"],
    "upi": ["KB002"],
    "payment": ["KB002"],
    "debit": ["KB002"],
    "account": ["KB003"],
    "suspend": ["KB003"],
    "login": ["KB003"],
    "lock": ["KB003"],
    "subscription": ["KB004"],
    "plan": ["KB004"],
    "upgrade": ["KB004"],
    "crash": ["KB005"],
    "app": ["KB005"],
    "technical": ["KB005"],
}

KB_MAP: Dict[str, KBArticle] = {a.article_id: a for a in KB}


def search_kb(query: str) -> List[KBArticle]:
    """Keyword-based KB lookup. Returns up to 3 relevant articles."""
    hits: Dict[str, KBArticle] = {}
    for word in query.lower().split():
        for aid in KB_INDEX.get(word, []):
            hits[aid] = KB_MAP[aid]
    return list(hits.values())[:3] or [KB_MAP["KB001"]]  # fallback


# ---------------------------------------------------------------------------
# Sample Tickets
# ---------------------------------------------------------------------------

TICKETS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "ticket_id": "T-EASY-01",
        "subject": "Charged twice for subscription",
        "description": (
            "Hi, I was charged ₹499 twice for my monthly subscription on 5th April. "
            "Please refund the duplicate charge immediately. "
            "My order ID is ORD-2024-9981."
        ),
        "gt_category": "billing",
        "gt_priority": "normal",
        "gt_action": "resolve",
        "conversation_history": [],
    },

    "medium": {
        "ticket_id": "T-MED-01",
        "subject": "UPI payment debited but order not placed",
        "description": (
            "I paid ₹1,299 via UPI (ref: UPI98765) for a yearly plan upgrade "
            "but my account still shows the free tier. "
            "It has been 2 days — please check."
        ),
        "gt_category": "billing",
        "gt_priority": "high",
        "gt_action": "resolve",
        "conversation_history": [
            Message(role="customer", content="I paid ₹1,299 two days ago and account not upgraded."),
        ],
    },

    "hard": {
        "ticket_id": "T-HARD-01",
        "subject": "URGENT: Fraud charge of ₹12,000 — demand immediate refund",
        "description": (
            "Someone made an unauthorized transaction of ₹12,000 on my account. "
            "I never approved this. I want a FULL refund NOW or I will file a complaint "
            "with the consumer forum. This is absolutely unacceptable!"
        ),
        "gt_category": "billing",
        "gt_priority": "urgent",
        "gt_action": "escalate",          # must escalate — amount > ₹5,000
        "escalate_dept": "fraud_team",
        "conversation_history": [
            Message(role="customer", content="I see a charge of ₹12,000 I never made."),
            Message(role="agent", content="We're sorry to hear this. Can you share the transaction date?"),
            Message(role="customer", content="It's 6th April. I want this fixed TODAY."),
            Message(role="agent", content="We are investigating. Please bear with us."),
            Message(role="customer", content="This is ridiculous! How long will it take?"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Per-difficulty config
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIG = {
    "easy":   {"max_steps": 6,  "kb_required": False, "clarification_allowed": False},
    "medium": {"max_steps": 8,  "kb_required": True,  "clarification_allowed": True},
    "hard":   {"max_steps": 12, "kb_required": True,  "clarification_allowed": False},
}

VALID_CATEGORIES = {"billing", "refund", "technical", "account", "shipping", "general"}
VALID_PRIORITIES = {"low", "normal", "high", "urgent"}


# ---------------------------------------------------------------------------
# SupportEnv
# ---------------------------------------------------------------------------

class SupportEnv:
    """
    OpenEnv-compliant customer support ticket resolver.

    Usage:
        env = SupportEnv()
        obs = env.reset("easy")
        obs, reward, done, info = env.step(action)
        score = env.grade()
    """

    def __init__(self) -> None:
        self._difficulty: str = "easy"
        self._obs: Optional[Observation] = None
        self._ticket_data: Dict[str, Any] = {}
        self._rewards: List[float] = []
        self._done: bool = False

        # Tracking flags for grader
        self._classified: bool = False
        self._kb_searched: bool = False
        self._clarification_asked: bool = False
        self._response_drafted: bool = False
        self._resolved: bool = False
        self._escalated: bool = False
        self._escalated_dept: str = ""
        self._action_history: List[str] = []

        self.reset("easy")

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, difficulty: str = "easy") -> Observation:
        """Start a new episode for the given difficulty level."""
        assert difficulty in DIFFICULTY_CONFIG, f"Unknown difficulty: {difficulty}"
        self._difficulty = difficulty
        td = copy.deepcopy(TICKETS[difficulty])
        self._ticket_data = td

        self._classified = False
        self._kb_searched = False
        self._clarification_asked = False
        self._response_drafted = False
        self._resolved = False
        self._escalated = False
        self._escalated_dept = ""
        self._action_history = []
        self._rewards = []
        self._done = False

        self._obs = Observation(
            ticket_id=td["ticket_id"],
            subject=td["subject"],
            description=td["description"],
            category=None,
            priority=None,
            status="open",
            conversation_history=list(td.get("conversation_history", [])),
            internal_notes=[],
            kb_results=[],
            available_actions=list(DIFFICULTY_CONFIG[difficulty].keys()) + [
                "classify_ticket", "search_kb", "ask_clarification",
                "draft_response", "resolve", "escalate", "add_internal_note",
            ],
            step_count=0,
        )
        return self._obs

    def state(self) -> Observation:
        """Return current observation without advancing the episode."""
        assert self._obs is not None, "Call reset() first."
        return self._obs

    def step(self, action: Any) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Args:
            action: A parsed Action model (ClassifyTicket, SearchKB, etc.)

        Returns:
            (observation, reward, done, info)
        """
        assert self._obs is not None, "Call reset() first."
        assert not self._done, "Episode already finished. Call reset()."

        cfg = DIFFICULTY_CONFIG[self._difficulty]
        self._obs.step_count += 1
        self._obs.last_action_error = None

        reward_val, reason = self._apply_action(action, cfg)

        # Step-count penalty to encourage efficiency
        reward_val -= 0.01

        reward_val = max(-1.0, min(1.0, reward_val))
        reward = Reward(value=round(reward_val, 3), reason=reason)
        self._rewards.append(reward.value)

        # Episode ends when resolved/escalated OR step budget exceeded
        if self._obs.status in ("resolved", "escalated", "closed"):
            self._done = True
        elif self._obs.step_count >= cfg["max_steps"]:
            self._done = True
            self._obs.status = "timeout"

        info = {
            "score": self.grade(),
            "difficulty": self._difficulty,
            "step_count": self._obs.step_count,
        }
        return self._obs, reward, self._done, info

    def grade(self) -> float:
        """
        Compute final score in [0.0, 1.0].
        Can be called at any time; reflects progress so far.
        """
        d = self._difficulty
        td = self._ticket_data

        score = 0.0

        # --- Category / Priority (shared across all difficulties) ---
        if self._obs.category == td["gt_category"]:
            score += 0.25
        elif self._obs.category in VALID_CATEGORIES:
            score += 0.05   # partial — at least classified

        if self._obs.priority == td["gt_priority"]:
            score += 0.20
        elif self._obs.priority in VALID_PRIORITIES:
            score += 0.05

        # --- Difficulty-specific scoring ---
        if d == "easy":
            # Must resolve
            if self._obs.status == "resolved":
                score += 0.30
            if self._response_drafted:
                score += 0.15
            # Efficiency bonus
            if self._obs.status == "resolved" and self._obs.step_count <= 3:
                score += 0.10

        elif d == "medium":
            # KB usage required
            if self._kb_searched:
                score += 0.15
            if self._response_drafted:
                score += 0.20
            if self._obs.status == "resolved":
                score += 0.15
            # Penalise if > 1 clarification
            if self._clarification_asked:
                clarif_count = self._action_history.count("ask_clarification")
                if clarif_count == 1:
                    score += 0.0   # neutral
                elif clarif_count > 1:
                    score -= 0.10  # too many clarifications

        elif d == "hard":
            # Must escalate (fraud > ₹5,000 policy)
            gt_action = td.get("gt_action", "resolve")
            gt_dept = td.get("escalate_dept", "")
            if gt_action == "escalate":
                if self._escalated:
                    score += 0.30
                    if self._escalated_dept == gt_dept:
                        score += 0.10   # correct department
                else:
                    score -= 0.20       # policy violation: did not escalate
            # KB consultation
            if self._kb_searched:
                score += 0.10
            # Internal note (shows diligence)
            if "add_internal_note" in self._action_history:
                score += 0.05

        # Efficiency bonus — finish in ≤ half the max steps
        max_steps = DIFFICULTY_CONFIG[d]["max_steps"]
        if self._done and self._obs.step_count <= max_steps // 2:
            score += 0.05

        return round(min(max(score, 0.0), 1.0), 3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: Any, cfg: Dict[str, Any]) -> Tuple[float, str]:
        """Dispatch action and return (reward_delta, reason_string)."""
        atype = getattr(action, "action_type", None)
        if atype is None:
            self._obs.last_action_error = "invalid action: missing action_type"
            return -0.20, "invalid action"

        # Repeated action penalty
        if self._action_history.count(atype) >= 2:
            self._obs.last_action_error = f"repeated action: {atype}"
            return -0.10, f"repeated {atype}"

        self._action_history.append(atype)

        if atype == "classify_ticket":
            return self._handle_classify(action)
        elif atype == "search_kb":
            return self._handle_search_kb(action)
        elif atype == "ask_clarification":
            return self._handle_ask_clarification(action, cfg)
        elif atype == "draft_response":
            return self._handle_draft_response(action)
        elif atype == "resolve":
            return self._handle_resolve(action)
        elif atype == "escalate":
            return self._handle_escalate(action)
        elif atype == "add_internal_note":
            return self._handle_internal_note(action)
        else:
            self._obs.last_action_error = f"unknown action_type: {atype}"
            return -0.20, "unknown action"

    def _handle_classify(self, action: ClassifyTicket) -> Tuple[float, str]:
        cat = action.category.lower()
        pri = action.priority.lower()
        if cat not in VALID_CATEGORIES:
            self._obs.last_action_error = f"invalid category: {cat}"
            return -0.10, "invalid category"
        if pri not in VALID_PRIORITIES:
            self._obs.last_action_error = f"invalid priority: {pri}"
            return -0.10, "invalid priority"

        self._obs.category = cat
        self._obs.priority = pri
        self._classified = True

        td = self._ticket_data
        reward = 0.0
        reason_parts = []
        if cat == td["gt_category"]:
            reward += 0.20
            reason_parts.append("correct category")
        else:
            reward -= 0.05
            reason_parts.append("wrong category")
        if pri == td["gt_priority"]:
            reward += 0.15
            reason_parts.append("correct priority")
        else:
            reward -= 0.05
            reason_parts.append("wrong priority")
        return reward, ", ".join(reason_parts)

    def _handle_search_kb(self, action: SearchKB) -> Tuple[float, str]:
        results = search_kb(action.query)
        self._obs.kb_results = results
        self._kb_searched = True
        reward = 0.15 if results else 0.05
        return reward, f"KB search returned {len(results)} result(s)"

    def _handle_ask_clarification(
        self, action: AskClarification, cfg: Dict[str, Any]
    ) -> Tuple[float, str]:
        if not cfg["clarification_allowed"]:
            self._obs.last_action_error = "clarification not allowed on this difficulty"
            return -0.15, "clarification not allowed"
        if self._clarification_asked:
            return -0.10, "already asked clarification"
        self._clarification_asked = True
        self._obs.conversation_history.append(
            Message(role="agent", content=action.question)
        )
        return 0.05, "clarification asked"

    def _handle_draft_response(self, action: DraftResponse) -> Tuple[float, str]:
        if len(action.content.strip()) < 20:
            return -0.05, "draft too short"
        self._response_drafted = True
        self._obs.conversation_history.append(
            Message(role="agent", content=action.content)
        )
        # Bonus if KB was consulted first
        bonus = 0.05 if self._kb_searched else 0.0
        return 0.20 + bonus, "response drafted" + (" with KB context" if bonus else "")

    def _handle_resolve(self, action: Resolve) -> Tuple[float, str]:
        td = self._ticket_data
        # Hard: resolving instead of escalating is a policy violation
        if self._difficulty == "hard" and td.get("gt_action") == "escalate":
            self._obs.last_action_error = "policy violation: should escalate, not resolve"
            return -0.30, "policy violation: resolve instead of escalate"
        self._obs.status = action.status
        self._resolved = True
        if action.final_notes:
            self._obs.internal_notes.append(action.final_notes)
        reward = 0.25 if self._response_drafted else 0.10
        return reward, f"ticket {action.status}"

    def _handle_escalate(self, action: Escalate) -> Tuple[float, str]:
        td = self._ticket_data
        # If escalation is not the expected action, penalise
        if td.get("gt_action") != "escalate":
            self._obs.last_action_error = "unnecessary escalation"
            return -0.15, "unnecessary escalation"
        self._obs.status = "escalated"
        self._escalated = True
        self._escalated_dept = action.department
        self._obs.internal_notes.append(f"Escalated to {action.department}: {action.reason}")
        dept_correct = action.department == td.get("escalate_dept", "")
        reward = 0.35 if dept_correct else 0.20
        return reward, "correct escalation" if dept_correct else "escalated wrong dept"

    def _handle_internal_note(self, action: AddInternalNote) -> Tuple[float, str]:
        self._obs.internal_notes.append(action.note)
        return 0.05, "internal note added"