"""
Picks who answers each question: Claude (claude_llm.py) on this Mac's Claude
login, or Groq (llm.py) — when LLM_PROVIDER=groq, when Claude can't be set
up, or for a while after Claude fails to answer.
"""

import os
import time

from config import LLM_PROVIDER, CLAUDE_MODEL, CLAUDE_EFFORT, CLAUDE_RETRY_SECS
import llm

_claude = None
_claude_out_until = 0.0     # time.monotonic() before which Groq answers instead


def _have_groq() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def start():
    """Set up the provider and say in the log who's answering."""
    global _claude
    if LLM_PROVIDER == "claude":
        try:
            import claude_llm
            login = claude_llm.login_status()
            _claude = claude_llm.ClaudeBrain()
            _claude.prewarm()
            fallback = "Groq if Claude can't answer" if _have_groq() else "no fallback (no GROQ_API_KEY)"
            print(f"[LLM] Claude {CLAUDE_MODEL} at {CLAUDE_EFFORT} effort, on {login}; {fallback}")
            return
        except Exception as exc:
            # ImportError included: claude-agent-sdk not installed.
            print(f"[LLM] Claude isn't available ({exc}) — using Groq")
    elif LLM_PROVIDER != "groq":
        print(f'[LLM] Unknown LLM_PROVIDER "{LLM_PROVIDER}" — using Groq')
    if not _have_groq():
        print("[LLM] GROQ_API_KEY isn't set either — she can't answer questions")
    else:
        print("[LLM] Groq")


def reply_chunks(conversation: list):
    """Yield the reply to the newest message in pieces, ready to speak.

    Claude first, then Groq — but only when Claude failed before saying
    anything; after that, a half-spoken answer can't be restarted, so the
    error goes to the caller.
    """
    global _claude_out_until
    if _claude is not None and (time.monotonic() >= _claude_out_until or not _have_groq()):
        spoke = False
        try:
            for chunk in _claude.stream(conversation):
                spoke = True
                yield chunk
            return
        except Exception as exc:
            if spoke or not _have_groq():
                raise
            _claude_out_until = time.monotonic() + CLAUDE_RETRY_SECS
            print(f"  [LLM] Claude couldn't answer ({exc}) — Groq answers for the next "
                  f"{CLAUDE_RETRY_SECS // 60} min")
    yield llm.ask_llm(conversation)
