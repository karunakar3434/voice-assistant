#!/usr/bin/env python3
"""
Voice Assistant "Almighty" — Wake phrase: "Hey Almighty"
Stack: faster-whisper (local STT) | Claude Sonnet via the Agent SDK, Groq
gpt-oss-120b as the fallback (LLM) | macOS say (TTS)

This file is just the main loop — everything else lives in its own module:
    config.py     — constants, wake words, thresholds, the persona/system prompt
    memory.py     — persistent conversation history + instructions
    tools.py      — tool schemas the LLM can call + what they actually do
    audio.py      — mic capture, Whisper transcription, TTS playback + barge-in
    brain.py      — picks Claude or Groq for each answer
    claude_llm.py — Claude through the Agent SDK, on this Mac's Claude login
    llm.py        — the Groq tool-calling loop

Capabilities:
  - Web search (Claude's WebSearch, or DuckDuckGo with Groq)
  - Local file / directory access
  - Persistent conversation memory across sessions
  - Continuous conversation after single wake word

Usage:
    Managed as a launchd service — see run_almighty.sh and README.md.
    For a foreground run, use ./run_almighty.sh rather than this file: it's
    what loads GROQ_API_KEY and OWNER_NAME from .env.
"""

import re
import threading
import time
from faster_whisper import WhisperModel

from config import (
    SAMPLE_RATE, WHISPER_WAKE, WHISPER_STT, MEMORY_FILE, INSTRUCTIONS_FILE,
    WAKE_VARIANTS, WAKE_PROMPT, WAKE_MIN_LOGPROB, STT_PROMPT, SLEEP_PHRASES, STOP_PHRASES,
    MAX_UTTERANCE_CHARS,
    CONFIRM_YES, CONFIRM_NO, CONFIRM_MAX_SECS, WAKE_DEBUG,
)
from memory import load_history, save_history
from audio import MicBuffer, capture_utterance, transcribe, speak, peak_rms
import brain
from llm import DailyLimitReached
from tools import set_confirmer

# Deny-biased on purpose: no is checked first, and a reply matching neither
# list is treated as a no.
_YES_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in CONFIRM_YES) + r")\b")
_NO_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in CONFIRM_NO) + r")\b")

# Words that can sit around a sleep phrase and still leave it a sleep command
# ("okay, okay, um, could you go to sleep now, almighty"). Hesitations matter:
# without "um" in here, that exact sentence went to the model, which replied
# "Got it, I'll be here whenever you need me" while she stayed awake.
_SLEEP_FILLER = {"okay", "ok", "alright", "right", "please", "now", "hey", "almighty",
                 "all", "mighty", "omighty", "thanks", "thank", "you", "um", "uh",
                 "erm", "er", "hmm", "mm", "so", "well", "yeah", "yes", "just",
                 "can", "could", "would", "will"}


def is_sleep_command(text: str) -> bool:
    """True only when the whole utterance is a sleep phrase, give or take filler.

    This used to be a substring check, so any sentence with "sleep" in it —
    "how much sleep do I need?", "the baby's asleep" — sent her back to sleep
    mid-conversation instead of getting an answer.
    """
    words = [w for w in re.findall(r"[a-z']+", text.lower()) if w not in _SLEEP_FILLER]
    return " ".join(words) in SLEEP_PHRASES


# Longest first, so "hold on a second" is taken as one phrase, not "hold on"
# with "a second" left over.
_STOP_BY_LENGTH = sorted(STOP_PHRASES, key=len, reverse=True)


def is_stop_command(text: str) -> bool:
    """True when the whole utterance only asks her to stop talking — "wait",
    "okay, hold on a sec", "no, stop, stop" — give or take filler.

    "no" counts as filler here but not on its own: a bare "no" answers her
    last question, and the model should hear it.
    """
    words = [w for w in re.findall(r"[a-z']+", text.lower())
             if w not in _SLEEP_FILLER and w != "no"]
    rest = " ".join(words)
    if not rest:
        return False
    while rest:
        for phrase in _STOP_BY_LENGTH:
            if rest == phrase or rest.startswith(phrase + " "):
                rest = rest[len(phrase):].lstrip()
                break
        else:
            return False
    return True


def transcribe_request(audio_bytes: bytes, model) -> str:
    """What the owner asked. Whisper is primed so "F1" stops coming out as
    "of one" or "a fun", and the VAD keeps it from hearing the prompt in a
    hum. The measurements are next to STT_PROMPT in config.py."""
    return transcribe(audio_bytes, model, initial_prompt=STT_PROMPT, vad_filter=True)


# One user of the speaker and mic at a time. Claude runs her tools on another
# thread, so without this run_command's spoken confirmation could talk over a
# heads-up that's still playing, with both listening to the mic at once.
_voice = threading.Lock()


def respond(conversation: list, mic) -> None:
    """Answer the newest message out loud, a piece at a time as the model
    finishes each one, then save what she said. Cutting in stops the rest of
    the reply, and stops the model working on it."""
    said = []
    chunks = brain.reply_chunks(conversation)
    try:
        for chunk in chunks:
            said.append(chunk)
            try:
                with _voice:
                    cut_in = speak(chunk, mic, interruptible=True)
            except Exception as exc:
                # A `say`/audio glitch: the reply still gets saved below, it
                # just wasn't voiced.
                print(f"  Speak error: {exc}")
                continue
            if cut_in:
                # Drop straight back to capturing what they said.
                print("  [interrupted — listening…]")
                break
    except Exception as exc:
        # Broad on purpose: anything from an API error to an unexpected bug
        # in our own code should degrade to a spoken apology, not a silent
        # crash (launchd would just restart the process with no indication
        # anything failed). Once she's said part of an answer it can't be
        # restarted, so that part is kept and the rest is lost.
        print(f"  LLM error: {exc}")
        if not said:
            conversation.pop()
            # A daily limit isn't a fault: retrying won't help until it frees
            # up, so say that and when, not "please try again".
            with _voice:
                speak(exc.spoken() if isinstance(exc, DailyLimitReached)
                      else "Sorry, I had trouble with that. Please try again.", mic)
            return
    finally:
        chunks.close()
    conversation.append({"role": "assistant", "content": " ".join(said)})
    save_history(conversation)


def main():
    print("=" * 60)
    print("  Voice Assistant")
    print(f"  Memory file : {MEMORY_FILE}")
    print(f"  Instructions: {INSTRUCTIONS_FILE}")
    print("=" * 60)

    print("\nLoading Whisper models …")
    wake_model = WhisperModel(WHISPER_WAKE, device="cpu", compute_type="int8")
    stt_model  = WhisperModel(WHISPER_STT,  device="cpu", compute_type="int8")
    brain.start()   # logs an [LLM] line saying who answers

    conversation = load_history()
    print('Ready. Say "Hey Almighty" to activate.\n')

    # Opens the default input and follows it from then on, logging a `Mic:`
    # line each time it (re)binds — see MicBuffer in audio.py.
    mic = MicBuffer()

    def confirm_out_loud(command: str, reason: str) -> bool:
        """Ask the owner, out loud, before running a destructive command.

        Deliberately routed around the LLM: the answer comes off the
        microphone, so text injected into the model's context can never
        approve anything on the owner's behalf. Anything that is not a clear
        yes — silence, ambiguity, a second unclear reply — is a no.
        """
        spoken = command if len(command) <= 200 else command[:200] + ", and more"
        with _voice:
            speak(
                f"Careful — that would {reason}. The command is: {spoken}. "
                "Should I run it?",
                mic,
            )
            for attempt in range(2):
                answer = transcribe(capture_utterance(mic, max_secs=CONFIRM_MAX_SECS), stt_model)
                if not answer:
                    continue
                print(f'  [Confirm] heard: "{answer}"')
                if _NO_RE.search(answer):
                    speak("Okay, leaving it.", mic)
                    return False
                if _YES_RE.search(answer):
                    return True
                if attempt == 0:
                    speak("Sorry — was that a yes or a no?", mic)
            speak("I'll leave it, since I didn't get a clear yes.", mic)
            return False

    set_confirmer(confirm_out_loud)

    try:
        activated = False
        while True:
            if not activated:
                # ── Wait for wake word ─────────────────────────────────────
                t0  = time.time()
                # A wake phrase takes ~1.5 s — the 8 s cap keeps a running TV
                # from tying up the wake check for 30 s at a time.
                raw = capture_utterance(mic, max_secs=15, max_speech_secs=8)
                if not raw:
                    continue
                t1 = time.time()
                # Prompt, no temperature fallback and a confidence floor — the
                # measurements behind each are next to WAKE_PROMPT in config.py.
                text = transcribe(raw, wake_model, initial_prompt=WAKE_PROMPT,
                                  temperature=0.0, min_avg_logprob=WAKE_MIN_LOGPROB)
                t2 = time.time()
                if WAKE_DEBUG:
                    print(f'[Wake debug] {len(raw) / 2 / SAMPLE_RATE:.1f}s, '
                          f'peak RMS {peak_rms(raw):.0f}, transcribe {t2 - t1:.2f}s: "{text}"')
                if not text or not any(v in text for v in WAKE_VARIANTS):
                    continue
                print(
                    f'\n[Wake timing] capture={t1 - t0:.2f}s '
                    f'transcribe={t2 - t1:.2f}s total={t2 - t0:.2f}s'
                )
                if is_sleep_command(text):
                    # Wake word matched, but the same utterance was also a
                    # sleep command (e.g. "go to sleep Almighty") — she's
                    # already asleep, so stay that way instead of waking
                    # just to immediately announce going back to sleep.
                    print(f'[Ignored — already asleep] heard: "{text}"')
                    continue
                print(f'[Activated] heard: "{text}"')
                speak("Hi, Almighty here. I'm listening.", mic)
                activated = True

            else:
                # ── Continuous conversation ────────────────────────────────
                # Under _voice in case a confirmation is still asking — after
                # she's interrupted, the tool that asks can still be running.
                with _voice:
                    req_audio = capture_utterance(mic, max_secs=10)
                req_text  = transcribe_request(req_audio, stt_model)

                if not req_text:
                    continue   # silence/timeout — stay active

                if len(req_text) > MAX_UTTERANCE_CHARS:
                    req_text = req_text[:MAX_UTTERANCE_CHARS]

                print(f"  You: {req_text}")

                if is_sleep_command(req_text):
                    speak("Going to sleep. Say 'Hey Almighty' when you need me.", mic)
                    activated = False
                    print('\nSay "Hey Almighty" to activate again.\n')
                    continue

                if is_stop_command(req_text):
                    # She's already quiet — just keep listening.
                    print("  [stop — listening…]")
                    continue

                conversation.append({"role": "user", "content": req_text})
                respond(conversation, mic)

    except KeyboardInterrupt:
        print("\nShutting down — history saved.")
        save_history(conversation)
    finally:
        mic.stop()   # also closes the stream and PortAudio


if __name__ == "__main__":
    main()
