"""
Persistent conversation history, permanent user instructions, and the
system-prompt assembly that stitches them together for each LLM call.
"""

import os
import json
from datetime import datetime

from config import MEMORY_FILE, INSTRUCTIONS_FILE, BASE_SYSTEM, CONTEXT_HEAD, CONTEXT_TAIL


def load_history():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[Memory] Loaded {len(data)} messages from previous sessions.")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[Memory] Could not load history: {exc}. Starting fresh.")
    return []


def save_history(conversation):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[Memory] Warning — could not save history: {exc}")


def load_instructions():
    if os.path.exists(INSTRUCTIONS_FILE):
        try:
            with open(INSTRUCTIONS_FILE, "r", encoding="utf-8") as f:
                lines = [ln for ln in f if not ln.lstrip().startswith("#")]
            text = "".join(lines).strip()
            if text:
                return text
        except OSError:
            pass
    return ""


def current_time_line():
    return f"Current date/time: {datetime.now().strftime('%A, %B %d %Y %H:%M')}"


def build_system_prompt(with_time=True):
    # Claude's session is started before the question arrives (see
    # claude_llm.py), so it leaves the time out here and puts it in the
    # question instead — otherwise she'd answer with the time she started.
    instructions = load_instructions()
    parts = [BASE_SYSTEM]
    if instructions:
        parts.append(f"\nPermanent user instructions:\n{instructions}")
    if with_time:
        parts.append(f"\n{current_time_line()}")
    return "\n".join(parts)


def trim_for_api(conversation):
    total = CONTEXT_HEAD + CONTEXT_TAIL
    if len(conversation) <= total:
        return conversation
    head = conversation[:CONTEXT_HEAD]
    tail = conversation[-CONTEXT_TAIL:]
    marker = {
        "role": "user",
        "content": (
            "[System note: older conversation messages were trimmed to fit the "
            "context window. The full history is saved on disk. The messages above "
            "are from the very beginning; the messages below are the most recent.]"
        ),
    }
    return head + [marker] + tail
