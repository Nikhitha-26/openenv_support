"""
tests/test_env.py — Deterministic tests for all 3 difficulty levels.
"""
 
import pytest
from envs.support_env import SupportEnv
from models import (
    ClassifyTicket, SearchKB, DraftResponse, Resolve, Escalate,
)
 
 
# ── Easy ────────────────────────────────────────────────────────────────────
 
def test_easy_full_episode():
    env = SupportEnv()
    obs = env.reset("easy")
    assert obs.ticket_id == "T-EASY-01"
    assert obs.status == "open"
 
    # classify
    obs, r, done, info = env.step(ClassifyTicket(category="billing", priority="normal"))
    assert obs.category == "billing"
    assert obs.priority == "normal"
    assert r.value > 0
    assert not done
 
    # draft
    obs, r, done, info = env.step(DraftResponse(
        content="We have identified the duplicate charge and will process your refund within 5-7 business days."
    ))
    assert not done
 
    # resolve
    obs, r, done, info = env.step(Resolve(status="resolved", final_notes="Duplicate charge refund initiated."))
    assert done
    assert obs.status == "resolved"
    assert info["score"] >= 0.6
 
 
def test_easy_score_in_range():
    env = SupportEnv()
    env.reset("easy")
    env.step(ClassifyTicket(category="billing", priority="normal"))
    env.step(Resolve(status="resolved", final_notes="done"))
    score = env.grade()
    assert 0.0 <= score <= 1.0
 
 
# ── Medium ──────────────────────────────────────────────────────────────────
 
def test_medium_kb_search_rewarded():
    env = SupportEnv()
    env.reset("medium")
    env.step(ClassifyTicket(category="billing", priority="high"))
    obs, r, done, info = env.step(SearchKB(query="UPI payment refund"))
    assert len(obs.kb_results) > 0
    assert r.value > 0
 
 
def test_medium_full_episode():
    env = SupportEnv()
    obs = env.reset("medium")
    assert obs.ticket_id == "T-MED-01"
 
    env.step(ClassifyTicket(category="billing", priority="high"))
    env.step(SearchKB(query="UPI payment debit refund"))
    env.step(DraftResponse(
        content=(
            "Thank you for reaching out. UPI payments that fail typically reverse "
            "within 48 hours. Your account upgrade will reflect once the payment is "
            "confirmed. We are escalating this to our billing team for faster resolution."
        )
    ))
    obs, r, done, info = env.step(Resolve(status="resolved", final_notes="UPI dispute raised."))
    assert done
    assert info["score"] >= 0.5
 
 
# ── Hard ────────────────────────────────────────────────────────────────────
 
def test_hard_must_escalate():
    env = SupportEnv()
    env.reset("hard")
    env.step(ClassifyTicket(category="billing", priority="urgent"))
    env.step(SearchKB(query="refund fraud unauthorized charge"))
    obs, r, done, info = env.step(Escalate(
        department="fraud_team",
        reason="Unauthorized charge of ₹12,000 reported — exceeds refund policy limit of ₹5,000."
    ))
    assert done
    assert obs.status == "escalated"
    assert info["score"] >= 0.6
 
 
def test_hard_resolve_without_escalate_penalised():
    env = SupportEnv()
    env.reset("hard")
    env.step(ClassifyTicket(category="billing", priority="urgent"))
    obs, r, done, info = env.step(Resolve(status="resolved", final_notes="Resolved without escalation."))
    # Policy violation — score should be low
    assert info["score"] < 0.5
 
 
def test_hard_score_in_range():
    env = SupportEnv()
    env.reset("hard")
    env.step(ClassifyTicket(category="billing", priority="urgent"))
    env.step(Escalate(department="fraud_team", reason="Fraud case"))
    score = env.grade()
    assert 0.0 <= score <= 1.0
 