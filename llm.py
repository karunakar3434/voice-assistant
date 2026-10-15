"""
The Groq-backed reasoning loop: builds each request, drives the tool-calling
round trip, and recovers from the model's occasional invented-tool-call
failures instead of aborting the whole answer. When Groq's daily limit runs
out it moves to the fallback model, and when both are out it raises
DailyLimitReached so she can say so instead of a generic apology.

Groq answers when Claude can't, or always with LLM_PROVIDER=groq — brain.py
decides which. Claude's side is claude_llm.py.
"""

import math
import os
import re
import json
import time
import openai

from config import (GROQ_BASE_URL, GROQ_MODEL, GROQ_FALLBACK_MODEL,
                    MAX_TOOL_ROUNDS, FINAL_ANSWER_RETRIES, SPEECH_REMINDER)
from memory import build_system_prompt, trim_for_api
from tools import TOOLS, execute_tool

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Derived from TOOLS, never hand-written. This list previously named four
# tools while seven were declared, so any turn that hit the recovery path
# below silently told the model that write_file, create_docx and create_pdf
# did not exist for the rest of that turn.
_TOOL_NAMES = ", ".join(t["function"]["name"] for t in TOOLS)

# Repeated at the end of every request, after the newest message. The speaking
# rules live in the long system prompt, but each request also carries the last
# ten turns of history, older wordier replies that named sources included, and
# the model copied those over the rules: tested fresh it kept to them, live it
# gave a four-sentence Artemis update ending "In short…". The rules in last
# position win.
_SPEECH_REMINDER = {"role": "system", "content": SPEECH_REMINDER}


def _groq_client() -> "openai.OpenAI":
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and run: export GROQ_API_KEY=your_key_here"
        )
    # Groq's free tier allows 8,000 tokens a minute, and one question with a
    # few searches can use most of that — the only LLM error in the log so far
    # was a 429. The limit clears within seconds and the SDK waits as Groq's
    # retry-after asks, so 5 retries instead of the default 2 turns "Sorry, I
    # had trouble with that" into a few seconds' pause.
    return openai.OpenAI(base_url=GROQ_BASE_URL, api_key=api_key, max_retries=5)


class DailyLimitReached(Exception):
    """Both models have used up today's Groq allowance. `wait` is the seconds
    until the first of them can answer again."""

    def __init__(self, wait: float):
        self.wait = wait
        super().__init__(f"daily limit reached on {GROQ_MODEL} and {GROQ_FALLBACK_MODEL}, "
                         f"first one frees up in {wait / 60:.0f} min")

    def spoken(self) -> str:
        minutes = math.ceil(self.wait / 60)
        if minutes <= 1:
            when = "a minute or so"
        elif minutes < 90:
            when = f"about {minutes} minutes"
        else:
            when = f"about {round(minutes / 60)} hours"
        return (f"I've hit my daily limit with Groq, so I can't answer right now. "
                f"Try me again in {when}.")


# Groq's 429 for a daily limit reads "...on tokens per day (TPD): Limit
# 200000, Used 198801, Requested 2599. Please try again in 10m4.8s..." — or
# "requests per day (RPD)". The per-minute kind says "per minute" and clears
# in seconds, so the SDK's retries handle it.
_WAIT_RE = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")

# When each model's daily limit frees up (time.monotonic()), so a model that's
# out isn't asked again until then. Forgotten on restart, which costs one
# rejected request per model to find out again.
_out_until: dict[str, float] = {}


def _daily_limit_wait(exc: openai.RateLimitError) -> float | None:
    """Seconds until a daily-limit 429 clears, or None if it's the per-minute kind."""
    message = str(exc)
    if "per day" not in message:
        return None
    match = _WAIT_RE.search(message)
    if match is None:
        return 600.0                 # Groq has always said; guess if it ever doesn't
    hours, minutes, seconds = match.groups()
    return max(60.0, int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0))


def _create(client, **kwargs):
    # The SDK retries a 429 five times with backoff, 11–15 s in all, and a
    # daily limit never clears in that time — she'd sit silent, then fail. So
    # try once without retries, and hand the request to the SDK's retries
    # only when the first try failed for a reason a short wait can fix.
    try:
        return client.with_options(max_retries=0).chat.completions.create(**kwargs)
    except openai.RateLimitError as exc:
        if _daily_limit_wait(exc) is not None:
            raise
    except (openai.APIConnectionError, openai.InternalServerError):
        pass
    return client.chat.completions.create(**kwargs)


def _complete(client, **kwargs):
    """One chat completion from GROQ_MODEL, or from GROQ_FALLBACK_MODEL while
    GROQ_MODEL's daily limit is used up. Raises DailyLimitReached when both are."""
    models = (GROQ_MODEL, GROQ_FALLBACK_MODEL)
    for model in models:
        if time.monotonic() < _out_until.get(model, 0):
            continue
        try:
            return _create(client, model=model, **kwargs)
        except openai.RateLimitError as exc:
            wait = _daily_limit_wait(exc)
            if wait is None:
                raise
            _out_until[model] = time.monotonic() + wait
            print(f"  [Daily limit] {model} is out for {wait / 60:.0f} min")
    raise DailyLimitReached(min(_out_until[m] for m in models) - time.monotonic())


def ask_llm(conversation: list) -> str:
    client   = _groq_client()
    messages = [{"role": "system", "content": build_system_prompt()}]
    messages += trim_for_api(conversation)
    messages.append(_SPEECH_REMINDER)

    base_kwargs = dict(
        # NOT a brevity lever: for a reasoning model, hidden "thinking"
        # tokens are billed against this same budget, so a small cap can
        # truncate mid-reasoning or right at the start of the visible
        # answer — regardless of how long that answer would've been (this
        # is what actually caused an ugly mid-word cutoff when this was
        # tried at 400). Brevity is enforced entirely via the system
        # prompt instead; keep this generous.
        max_tokens  = 1024,
        temperature = 0.75,
        messages    = messages,
        # gpt-oss supports tool use natively alongside reasoning, unlike the
        # qwen model this replaced (which needed reasoning_effort "none" to
        # avoid looping on tool calls with empty content). "high" broke on
        # multi-search questions (irrelevant tool call, then empty final
        # answer — a known gpt-oss-120b + tool-calling instability on Groq
        # at high effort). "low" was rock solid but did minimal research
        # (one search, done). "medium" is the middle ground: noticeably
        # more thorough (multi-round research) while still landing cleanly
        # in testing — watch for a repeat of high's failure mode, since
        # it's been verified stable fewer times than low.
        extra_body  = {"reasoning_effort": "medium"},
    )

    for round_no in range(MAX_TOOL_ROUNDS):
        # On the final allowed round, force a spoken answer (no more tools).
        force_answer = round_no == MAX_TOOL_ROUNDS - 1

        request_kwargs = dict(base_kwargs)
        if not force_answer:
            request_kwargs["tools"] = TOOLS
            request_kwargs["tool_choice"] = "auto"
        # On the forced-answer round, omit `tools` entirely rather than just
        # setting tool_choice="none" — at high reasoning effort, gpt-oss-120b
        # sometimes tries to call a tool anyway when mid-research, and Groq's
        # API then hard-rejects the whole response with a 400 ("Tool choice
        # is none, but model called a tool"). Hiding the tools makes that
        # rarer but not impossible — it still happens mid-research, so the
        # loop below this one picks up the pieces.

        try:
            response = _complete(client, **request_kwargs)
        except openai.BadRequestError as exc:
            # gpt-oss-120b has a baked-in habit (from its own training, not
            # tied to reasoning_effort) of trying to "open" URLs it finds in
            # search results using invented tools we never declared
            # (open_file, web_open, ...). Groq rejects the whole response
            # when that happens. Recover instead of aborting the answer:
            # tell it what actually exists and let it retry.
            print(f"  [Tool-call recovery] {exc}")
            messages.append({
                "role": "system",
                "content": (
                    "Your last action referenced a tool that doesn't exist. "
                    f"Only these tools are available: {_TOOL_NAMES}. Use one "
                    "of them if you still need information, otherwise answer "
                    "directly with what you already have."
                ),
            })
            continue

        msg = response.choices[0].message

        if msg.tool_calls and not force_answer:
            # Echo the assistant turn (with its tool calls) back into the thread.
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"  [Tool] {tc.function.name} ← {args}")
                result = execute_tool(tc.function.name, args)
                messages.append({
                    "role"         : "tool",
                    "tool_call_id" : tc.id,
                    "content"      : str(result),
                })
            continue

        # Defensive: strip any reasoning tags in case a model leaks them.
        text = _THINK_RE.sub("", msg.content or "").strip()
        if text:
            return text

    # Out of rounds without an answer. Weather questions all ended here: four
    # searches that never turned up a live temperature, then a tool call on
    # the tool-less last round, which Groq rejected — and she said "Sorry, I
    # couldn't put together an answer". Say plainly that the tools are gone
    # and ask again, so she at least says what she found or that she couldn't.
    for _ in range(FINAL_ANSWER_RETRIES):
        messages.append({
            "role": "system",
            "content": (
                "No more tools — you can't search or run anything now. Answer "
                "with what you've already found. If you didn't find what was "
                "asked, say so in a sentence and suggest where they could check."
            ),
        })
        try:
            response = _complete(client, **base_kwargs)
        except openai.BadRequestError as exc:
            print(f"  [Tool-call recovery] {exc}")
            continue
        text = _THINK_RE.sub("", response.choices[0].message.content or "").strip()
        if text:
            return text

    # Never leave the user with silence.
    return "Sorry, I couldn't put together an answer for that. Could you rephrase?"
