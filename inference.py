"""
inference.py — OpenEnv Support Ticket Resolver: LLM Agent Inference Script
===========================================================================

Environment variables (required before running):
    API_BASE_URL   The LLM API base URL  (default: HuggingFace router)
    MODEL_NAME     Model identifier       (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key / HF token
    TASK           Task difficulty: easy | medium | hard | all  (default: all)

Stdout format (strictly enforced — do NOT change):
    [START] task=<name> env=<bench> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from envs.support_env import SupportEnv
from models import (
    AddInternalNote, AskClarification, ClassifyTicket,
    DraftResponse, Escalate, Resolve, SearchKB,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.getenv("HF_TOKEN", "")
TASK: str         = os.getenv("TASK", "all")   # easy | medium | hard | all

BENCHMARK = "support-ticket-resolver-openenv"
MAX_STEPS = 10
TEMPERATURE = 0.2   # low temperature for deterministic structured output
MAX_TOKENS  = 300
SUCCESS_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Stdout logging helpers  (field order is mandatory — do not change)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitise action string (no newlines)
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support agent AI. You handle support tickets step by step.

At each step you must output ONLY a valid JSON object representing ONE action.
Do NOT add any explanation, markdown, or extra text — only raw JSON.

Available action types and their fields:
  {"action_type": "classify_ticket", "category": "<billing|refund|technical|account|shipping|general>", "priority": "<low|normal|high|urgent>"}
  {"action_type": "search_kb", "query": "<search query string>"}
  {"action_type": "ask_clarification", "question": "<question for customer>"}
  {"action_type": "draft_response", "content": "<response text to customer>"}
  {"action_type": "resolve", "status": "resolved", "final_notes": "<brief note>"}
  {"action_type": "escalate", "department": "<billing_team|senior_support|fraud_team|technical_team>", "reason": "<reason>"}
  {"action_type": "add_internal_note", "note": "<internal note text>"}

Guidelines:
- Always classify the ticket first (category + priority).
- For billing/refund issues involving large amounts (>5000), you MUST escalate to fraud_team.
- Search the KB before drafting any customer response.
- Keep responses professional, empathetic, and concise.
- Resolve or escalate as soon as you have enough information.
- Output only one JSON action per turn.
""").strip()


def build_user_prompt(obs_dict: Dict[str, Any], step: int) -> str:
    """Build the per-step user message from the observation."""
    history = obs_dict.get("conversation_history", [])
    history_str = "\n".join(
        f"  [{m['role'].upper()}]: {m['content']}" for m in history
    ) or "  (none)"

    kb = obs_dict.get("kb_results", [])
    kb_str = "\n".join(
        f"  [{a['article_id']}] {a['title']}: {a['snippet']}" for a in kb
    ) or "  (not searched yet)"

    notes = obs_dict.get("internal_notes", [])
    notes_str = "\n".join(f"  - {n}" for n in notes) or "  (none)"

    return textwrap.dedent(f"""
Step {step} — ticket state:
  ID:          {obs_dict['ticket_id']}
  Subject:     {obs_dict['subject']}
  Description: {obs_dict['description']}
  Category:    {obs_dict.get('category') or 'not classified'}
  Priority:    {obs_dict.get('priority') or 'not set'}
  Status:      {obs_dict.get('status', 'open')}

Conversation history:
{history_str}

KB results (from your last search):
{kb_str}

Internal notes:
{notes_str}

Error from last action: {obs_dict.get('last_action_error') or 'none'}

Output your next action as a single JSON object now.
""").strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_llm_action(
    client: OpenAI,
    obs_dict: Dict[str, Any],
    step: int,
    history: List[Dict[str, str]],
) -> str:
    """Call the LLM and return the raw text response."""
    user_content = build_user_prompt(obs_dict, step)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep last 4 exchanges for context (avoids token bloat)
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

ACTION_MODELS = {
    "classify_ticket":  ClassifyTicket,
    "search_kb":        SearchKB,
    "ask_clarification": AskClarification,
    "draft_response":   DraftResponse,
    "resolve":          Resolve,
    "escalate":         Escalate,
    "add_internal_note": AddInternalNote,
}


def parse_action(raw_text: str):
    """
    Parse the LLM output into an Action model.
    Falls back to a safe classify action on failure.
    """
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()

    try:
        data = json.loads(text)
        atype = data.get("action_type")
        if atype in ACTION_MODELS:
            return ACTION_MODELS[atype].model_validate(data), None
        return None, f"unknown action_type: {atype}"
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, difficulty: str) -> None:
    """Run one complete episode and emit [START], [STEP]*, [END]."""
    task_name = f"{difficulty}_triage"
    env = SupportEnv()
    obs = env.reset(difficulty)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    llm_history: List[Dict[str, str]] = []

    try:
        for step in range(1, MAX_STEPS + 1):
            obs_dict = obs.model_dump()
            raw_text = get_llm_action(client, obs_dict, step, llm_history)

            action, parse_error = parse_action(raw_text)
            if action is None:
                # Fallback: classify as billing/normal to keep the episode moving
                action = ClassifyTicket(category="billing", priority="normal")
                parse_error = parse_error or "parse failed, used fallback"

            action_str = json.dumps(action.model_dump(), ensure_ascii=False)

            obs, reward, done, info = env.step(action)
            error_out = obs.last_action_error or parse_error

            rewards.append(reward.value)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward.value,
                done=done,
                error=error_out,
            )

            # Track for LLM context
            llm_history.append({"role": "user", "content": build_user_prompt(obs_dict, step)})
            llm_history.append({"role": "assistant", "content": raw_text})

            if done:
                break

        score = info.get("score", env.grade())
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("[DEBUG] HF_TOKEN not set — proceeding (some endpoints may reject)", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    tasks = ["easy", "medium", "hard"] if TASK == "all" else [TASK]
    for difficulty in tasks:
        run_episode(client, difficulty)


if __name__ == "__main__":
    main()