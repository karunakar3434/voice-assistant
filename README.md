# Almighty — Personal Voice Assistant

An always-on voice assistant for macOS. The wake word and speech recognition run locally with faster-whisper, and answers come from Claude Sonnet on your Claude Pro or Max plan, through the Claude Agent SDK — with Groq's free gpt-oss-120b as the fallback, or on its own if you don't have a Claude plan.

## Developer

Built by **Karun** as a fully local, always-on personal voice assistant for macOS. Designed to run 24/7 as a background service, remember every conversation, and handle real-world tasks through natural speech — no cloud dependencies for audio processing, no third-party wake-word service, no paid API key — she runs on a Claude plan you already have, or on Groq's free tier.

---

## What is Almighty?

Almighty is a voice-activated AI research assistant that lives on your Mac as a background service. Say **"Hey Almighty"** once to activate her, then have a continuous back-and-forth conversation — no need to repeat the wake phrase. She keeps every conversation across sessions, digs up and reasons through facts via live web search, controls your Mac's system settings, and creates documents on your behalf.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Wake word detection | faster-whisper (`base` model, local, CPU) |
| Speech-to-text | faster-whisper (`base` model, local, CPU) |
| Language model | Claude Sonnet 5.5 (`claude-sonnet-5-5`, `low` effort) on your Claude Pro or Max login. Groq's `openai/gpt-oss-120b` (`medium` effort, and `openai/gpt-oss-20b` while the 120b's daily limit is used up) answers whenever Claude can't, or always with `LLM_PROVIDER=groq` |
| LLM client | Claude Agent SDK (`claude-agent-sdk`, which brings its own copy of the Claude Code CLI) for Claude; the OpenAI Python SDK, pointed at Groq's endpoint, for Groq |
| Text-to-speech | macOS built-in `say` command (Samantha Enhanced voice) |
| Audio capture | PyAudio — 16 kHz, mono, paInt16 |
| Web search | DDGS (DuckDuckGo Search, no API key needed), with either model — Claude's built-in WebSearch is one setting away (`CLAUDE_SEARCH` in `config.py`). It was more accurate on live numbers like prices, but took twice as long |
| Weather | Open-Meteo geocoding and forecast APIs, called with `urllib` (no API key or extra package needed) |
| Document creation | python-docx (Word), ReportLab (PDF) |
| Runtime | Homebrew Python 3.11 (`/opt/homebrew/bin/python3.11`) |
| Process supervision | macOS `launchd` (LaunchAgent) — auto-starts at login, auto-restarts on crash |
| Platform | macOS (Apple Silicon) |

---

## Capabilities

### Conversation & Memory
- Activates on the wake phrase **"Hey Almighty"** (also matches "all mighty", "all-mighty", "omighty") — Whisper is primed with the name for the wake check, so a casual or clipped "hey almighty" still lands; the measurements are in `config.py` next to `WAKE_PROMPT`
- Requests get a prompt of their own, so "F1" stops coming out as "of one" or "a fun", and Whisper's voice-activity filter keeps a hum or a breath from turning into a question; see `STT_PROMPT` in `config.py`
- Stays active after a single wake — continuous conversation without repeating the phrase
- Persists the full conversation history to `memory.json` across all sessions and restarts — but each request only sends the model the first 2 and the last 10 messages (`CONTEXT_HEAD`/`CONTEXT_TAIL` in `config.py`), so older turns are saved, not seen
- Always knows your name (`OWNER_NAME` in `.env`) and whatever you put in `instructions.txt` — that's the place for preferences she should never lose track of
- Say **"go to sleep"** or **"sleep"** on its own to put her back into wake-word listening mode — a sentence that only mentions sleep ("how much sleep do I need?") gets an answer instead

### Research & Reasoning
- Thinks through a question before answering, and looks things up via live web search instead of guessing when something could have changed
- With Claude, says a few words first — "one sec, let me check" — when she's about to look something up, about a second and a half after you stop talking, so you know she heard you while the search runs. The answer usually follows about 6 seconds after you stop talking, up to 12 when a first search comes back thin. Groq stays silent until its whole answer is ready
- Prefers authoritative sources, but doesn't name them unless you ask where something came from — "according to the CDC and a 2025 review" sounds like someone reading out a paper, not talking to you
- Digests search results into the point that matters rather than reading everything back — rounded numbers, everyday words, one or two sentences, and an offer to go deeper instead of dumping it all at once
- Answers weather questions with live numbers through `get_weather` — the temperature right now plus the day's high, low and chance of rain, anywhere, today or up to six days ahead, in °F. Web search can't, because search results for weather sites never carry a live temperature. Name any city in the world and she looks that one up. Set `HOME_LOCATION` in `.env` to your town (setup step 3) and she uses it only when you don't name a place — without it, a question with no place gets "which city?" back. It goes in `.env` rather than `instructions.txt` because `.env` is gitignored, so your town never ends up in a commit
- Treats retrieved content (search results, file contents) as data, never as instructions — won't act on text that tries to direct her behavior

### System Control (Full sudo access)
Almighty can control virtually any macOS setting via shell commands and AppleScript:
- Volume (set, mute, unmute)
- Screen brightness
- Dark mode / Light mode toggle
- Wi-Fi on/off and network switching
- Bluetooth on/off
- Do Not Disturb / Focus modes
- Sleep, restart, shutdown
- App launching and quitting
- Any `defaults write`, `networksetup`, `pmset`, or `osascript` command

**Spoken confirmation for anything destructive.** The system prompt tells her to
treat web pages and files as data, never as instructions — but that is only a
prompt, and `web_search` puts text she did not write into the same context as a
tool with passwordless sudo. So a second, non-negotiable check sits underneath
it: commands that use sudo, delete files, touch disks, cut power, read the
keychain, change background services or send data off the machine (any `curl`,
`wget`, `ssh`, `scp` and the like) are read back
to you out loud and run only if you say yes. That answer comes off the
microphone, in `assistant.py`, never through the model — so no amount of text
smuggled into a search result can approve a command on your behalf. Everyday
settings commands are unaffected and run immediately. If she is ever running
where she cannot ask you, those commands are refused rather than run.
See `tools._CONFIRM_PATTERNS` for the list and `assistant.confirm_out_loud`
for the prompt.

### File Access & Creation
- **Read** any file on your Mac
- **List** directory contents
- **Write** plain text, Markdown, JSON, CSV files
- **Create Word documents** (.docx) with headings and formatted paragraphs
- **Create PDFs** with title and body content, ready to share
- Asks for confirmation before overwriting an existing file rather than silently replacing it

### Barge-in
- Interrupt Almighty mid-sentence — she detects loud sound during playback, stops talking and keeps the start of what you said, so the interruption itself becomes your next request. With Claude it also stops her working on the rest of the answer
- Cut in with just **"stop"**, **"wait"**, **"hold on"** or **"never mind"** and she stays quiet and keeps listening — that never reaches the model, which used to answer "okay, wait, stop" with "I'm not sure what you mean by stop". The list is `STOP_PHRASES` in `config.py`. Anything more ("wait, who was the first emperor?") is a request like any other
- **Use headphones/AirPods for this to actually work well.** Without them, the Mac's mic can pick up Almighty's own voice from the speakers and mistake it for an interruption — there's no acoustic echo cancellation (a real attempt was made and reverted; see the comments above `speak()` in `audio.py` for the postmortem)

---

## Setup Instructions

### Prerequisites

| Requirement | How to get it |
|---|---|
| macOS (Apple Silicon) | — |
| Homebrew | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| Python 3.11 | `brew install python@3.11` |
| PortAudio (for mic) | `brew install portaudio` |
| Claude Pro or Max plan, signed in to Claude Code | `curl -fsSL https://claude.ai/install.sh \| bash`, then setup step 3 — skip it if she'll run on Groq only |
| Groq API key (free) | Sign up at [console.groq.com](https://console.groq.com) → API Keys — optional with Claude, where it's the fallback, and required with `LLM_PROVIDER=groq` |

### Installation

**1. Clone the repo into your home folder:**
```bash
git clone https://github.com/karunakar3434/voice-assistant.git ~/voice-assistant
```
Every command below assumes `~/voice-assistant`. If you clone somewhere else, swap your path in everywhere — including the `sed` command in step 5, or the service won't find its files.

**2. Install Python dependencies:**
```bash
/opt/homebrew/bin/python3.11 -m pip install -r ~/voice-assistant/requirements.txt
```
Use `python3.11` specifically, not whatever `python3` points to — `run_almighty.sh` starts `/opt/homebrew/bin/python3.11`, so packages installed into any other Python won't be there when the service starts. Install PortAudio first (see Prerequisites), or PyAudio fails to build.

**3. Sign in to Claude, then add your name and, optionally, a Groq key and your town:**
```bash
claude
```
Log in with your Pro or Max account when it asks, then type `/exit`. The login lands in your macOS keychain, where the copy of Claude Code inside the Agent SDK finds it too — the background service included, so there's no key to paste anywhere. Without a Claude plan, skip this and run her on Groq instead (below).
```bash
echo 'export OWNER_NAME="YourName"' > ~/voice-assistant/.env
echo 'export GROQ_API_KEY=your_key_here' >> ~/voice-assistant/.env
```
Put your name in place of `YourName` — the persona and the tool prompts address you by it. Leave it out and they just say "the user". The Groq key is optional with Claude: it's her fallback when Claude can't answer — a usage limit, signed out, no connection — and without it she apologises instead. Put your real key in place of `your_key_here`, or leave that line out. `.env` is gitignored, so the key never ends up in a commit, and `.env.example` has every line if you'd rather copy it.

No Claude plan? Add this too, and the Groq key becomes required — without it `run_almighty.sh` exits at start:
```bash
echo 'export LLM_PROVIDER=groq' >> ~/voice-assistant/.env
```

Optionally, add your town too:
```bash
echo 'export HOME_LOCATION="Austin, TX"' >> ~/voice-assistant/.env
```
Put your own town in place of `Austin, TX`, with its state or country — `get_weather` looks it up exactly as written, so a bare name can land on the wrong one of several towns. It only fills in when you don't name a place: "what's the weather doing tomorrow?" gets your town's forecast, while "what's it like in Tokyo?" still gets Tokyo's. Leave it out and a question with no place gets "which city?" back.

**4. (Optional) Set up passwordless sudo so Almighty can control system settings without prompting:**
```bash
echo "$(whoami) ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/almighty-nopasswd
sudo chmod 440 /etc/sudoers.d/almighty-nopasswd
sudo visudo -cf /etc/sudoers.d/almighty-nopasswd
```
The last line should print `parsed OK` — a malformed file in `/etc/sudoers.d/` breaks `sudo` for your whole account, so don't skip the check. Skip the whole step and everything else still works — commands that need sudo just fail, and she tells you so.

**5. Install the LaunchAgent so she runs as a 24/7 background service:**
```bash
chmod +x ~/voice-assistant/run_almighty.sh
sed "s|__PROJECT_DIR__|$HOME/voice-assistant|g" ~/voice-assistant/com.karun.almighty.plist > ~/Library/LaunchAgents/com.karun.almighty.plist
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.karun.almighty.plist
```
The plist in the repo has a `__PROJECT_DIR__` placeholder instead of a real path — launchd needs absolute paths and won't expand `~`, so `sed` fills in yours on the way in. If you put the project somewhere other than `~/voice-assistant`, change that part of the `sed` command too, or launchd won't find `run_almighty.sh` and she'll never start.

If `launchctl bootstrap` fails with `Bootstrap failed: 5: Input/output error`, the service is already loaded — run `launchctl bootout "gui/$(id -u)/com.karun.almighty"` and bootstrap again.

This starts her immediately and registers her to start automatically every time you log in. `run_almighty.sh` also keeps the Mac awake (via `caffeinate`) for as long as she's running, so she keeps listening instead of going idle — note this can't override a MacBook physically sleeping because the lid is closed with no external display attached, that's enforced in hardware.

**6. Check she's up:**
```bash
tail -f ~/voice-assistant/assistant.log
```
The first start downloads the Whisper `base` model (about 140 MB) from Hugging Face into `~/.cache/huggingface/`, so give it a minute. It needs internet — if the download fails she exits, and launchd keeps restarting her until it works. The `unauthenticated requests to the HF Hub` warning is harmless. She's ready when the log shows `Ready. Say "Hey Almighty" to activate.` — then say it. macOS may ask for microphone access the first time — allow it, or she hears silence and never wakes.

If she doesn't react, test the mic on its own:
```bash
/opt/homebrew/bin/python3.11 ~/voice-assistant/mic_test.py
```
It lists input devices and shows a live level meter. If the bars don't move when you talk, she can't hear you either — fix the input device in System Settings → Sound first.

### Running Almighty

She's meant to run as the background service above — you shouldn't need to start her manually. If you do want a one-off foreground run for debugging (**don't** do this while the LaunchAgent is also running — two instances will fight over the microphone):
```bash
~/voice-assistant/run_almighty.sh
```
That's the same script launchd runs, so it loads `.env` and keeps the Mac awake — the output just goes to your terminal instead of the log, and Ctrl+C stops her. Don't run `assistant.py` directly — nothing loads `.env` then, so she doesn't know your name or town and has no Groq fallback, and with `LLM_PROVIDER=groq` every answer fails with "Sorry, I had trouble with that."

### Personalising Almighty

Edit `instructions.txt` to give Almighty permanent instructions — changes take effect on the next activation with no restart needed:

```
# instructions.txt
Always respond in a casual, friendly tone.
My preferred document save location is ~/Documents.
```

Lines starting with `#` are ignored.

Your name and your town don't go here — they're `OWNER_NAME` and `HOME_LOCATION` in `.env` (setup step 3), which the persona in `config.py` and the tool prompts in `tools.py` are built from. Unlike `instructions.txt`, they're only read at startup, so restart the service after changing either:
```bash
launchctl kickstart -k "gui/$(id -u)/com.karun.almighty"
```

If you want the enhanced voice, download it in System Settings → Accessibility → Spoken Content → System voice → Manage Voices → English → Samantha (Enhanced). Without it, `say` quietly falls back to the default system voice — she still talks, just in a different voice.

### Managing the service

```bash
# Check she's running
launchctl print "gui/$(id -u)/com.karun.almighty" | grep state

# Restart her after a code change or a .env change
# (no need after switching mics or speakers — she follows the
# system defaults on her own)
launchctl kickstart -k "gui/$(id -u)/com.karun.almighty"

# Stop her (also stops her restarting at next login)
launchctl bootout "gui/$(id -u)/com.karun.almighty"

# Watch her live
tail -f ~/voice-assistant/assistant.log
```

Say **"go to sleep"** any time to put her back into wake-word listening mode without stopping the service.

### Running the tests

```bash
/opt/homebrew/bin/python3.11 -m unittest discover ~/voice-assistant/tests
```
They cover the sleep and yes/no matching, the spoken-confirmation gate, the weather tool (with the network mocked), utterance capture, transcription filtering, `say` argument handling, the Groq loop and the Claude session (both with the model faked), the switch between them, speaking a reply in pieces, memory and config — everything that doesn't need a microphone, a Whisper model, a Claude login or a Groq key, so they run in under a second and don't touch the running service. Run them after a change, before restarting her — a broken gate pattern or capture loop only shows up live as a command running unasked or a request cut short.

---

## File Reference

| File | Purpose |
|---|---|
| `assistant.py` | Entry point — the main wake/conversation loop |
| `config.py` | Constants: wake words, thresholds, model names, the persona/system prompt |
| `memory.py` | Persistent conversation history + permanent instructions |
| `tools.py` | Tool schemas the LLM can call (`web_search`, `get_weather`, `read_file`, `list_directory`, `run_command`, `write_file`, `create_docx`, `create_pdf`), and what they actually do |
| `audio.py` | Mic capture, Whisper transcription, TTS playback + barge-in |
| `brain.py` | Picks who answers each question — Claude, or Groq when Claude can't — and logs which at startup |
| `claude_llm.py` | Claude through the Agent SDK: the session kept ready between questions, her tools served in-process, the reply handed over a piece at a time |
| `llm.py` | The Groq tool-calling loop, and the switch to the fallback model when the daily limit runs out |
| `run_almighty.sh` | launchd entry point — loads `.env`, rotates the log, keeps the Mac awake, execs `assistant.py` |
| `com.karun.almighty.plist` | LaunchAgent template — step 5 fills in `__PROJECT_DIR__` and installs it into `~/Library/LaunchAgents/` |
| `memory.json` | Full persistent conversation history — created on first run, not in the repo (gitignored) |
| `instructions.txt` | Permanent instructions loaded every session |
| `requirements.txt` | Python package list |
| `mic_test.py` | Standalone microphone diagnostics / live energy-level meter — run manually, not used by the assistant itself |
| `tests/` | Unit tests — see "Running the tests" |
| `assistant.log` | Runtime log, rotated once it passes 5MB — `run_almighty.sh` only checks when she starts, so a run of several weeks can take it past that. Created at runtime, not in the repo (gitignored) |
| `.env.example` | Template for `.env` — `LLM_PROVIDER`, `GROQ_API_KEY`, `OWNER_NAME` and the rest — which is gitignored |
| `LICENSE` | MIT license |

---

## Local files (not in the repo)

A fresh clone doesn't have these — they're gitignored because they hold your API key, your conversations or machine-specific output. Only `.env` is yours to create. Everything else appears on its own once she runs.

| File | Created by | When |
|---|---|---|
| `.env` | You, in setup step 3 | Before the first start — without it she doesn't know your name, and with `LLM_PROVIDER=groq` `run_almighty.sh` exits and launchd keeps retrying |
| `memory.json` | `memory.py` | After her first reply |
| `assistant.log` | launchd (the plist's `StandardOutPath`/`StandardErrorPath`) | First start of the service |
| `assistant.log.old` | `run_almighty.sh` | When `assistant.log` passes 5 MB at a start — the previous log moves here |
| `__pycache__/` | Python | First import of each module |

Outside the project folder, setup also creates `~/Library/LaunchAgents/com.karun.almighty.plist` (step 5), `/etc/sudoers.d/almighty-nopasswd` (step 4, optional), Claude Code's login in your macOS keychain (step 3) and the Whisper model cache in `~/.cache/huggingface/hub/models--Systran--faster-whisper-base` (first start).

### `.env`

A few lines. `run_almighty.sh` sources it with `set -a`, so the `export` is optional. With Claude none of them is required; with `LLM_PROVIDER=groq` the Groq key is:
```bash
export LLM_PROVIDER=claude
export GROQ_API_KEY=your_key_here
export OWNER_NAME="YourName"
export HOME_LOCATION="Austin, TX"
```

### `memory.json`

A JSON array of every turn, oldest first. Each entry is `{"role": "user" | "assistant", "content": "..."}` — what you said (as Whisper transcribed it, lowercased) and what she said back. A turn is saved only once she's answered. The file keeps growing — only the first 2 and last 10 messages are sent to the model on each request (`CONTEXT_HEAD`/`CONTEXT_TAIL` in `config.py`).
```json
[
  {
    "role": "user",
    "content": "what's the tallest building in the world?"
  },
  {
    "role": "assistant",
    "content": "That's the Burj Khalifa in Dubai, about 828 meters tall."
  }
]
```
To make her forget everything, stop the service first, then delete the file — she holds the history in memory while running and rewrites the whole file on her next reply. A hand edit that breaks the JSON loses the history too — on her next start she can't load it, starts fresh and overwrites the file.

### `assistant.log`

Everything she prints, plus Python errors. A typical start and one exchange:
```
[2026-10-15 09:00:00] Starting Almighty (launchd-managed)...
============================================================
  Voice Assistant
  Memory file : /Users/<you>/voice-assistant/memory.json
  Instructions: /Users/<you>/voice-assistant/instructions.txt
============================================================

Loading Whisper models …
[LLM] Claude claude-sonnet-5-5 at low effort, on your Claude max login; Groq if Claude can't answer
[Memory] Loaded 2 messages from previous sessions.
Ready. Say "Hey Almighty" to activate.

Mic: Your AirPods Pro

[Wake timing] capture=1.42s transcribe=0.61s total=2.03s
[Activated] heard: "hey almighty."
  Assistant: Hi, Almighty here. I'm listening.
  You: what's the tallest building in the world?
  Assistant: One sec, let me check.
  [Tool] web_search ← {'query': 'tallest building in the world'}
  [Barge-in] peak mic RMS during playback: 190 (threshold 550)
  Assistant: That's the Burj Khalifa in Dubai, about 828 meters tall.
  [Barge-in] peak mic RMS during playback: 210 (threshold 550)
```
`[LLM]` says who's answering — if it says Claude isn't available, Troubleshooting has the fix. The `[Memory] Loaded …` line only appears once `memory.json` exists. `Mic:` names the input she's listening on — it's logged at startup and again within about a second of the default input changing, so check it first when she stops waking.

---

## Troubleshooting

| What you see | Why | Fix |
|---|---|---|
| Log repeats `ERROR: LLM_PROVIDER=groq but GROQ_API_KEY not set in .env` | You chose Groq only, and `.env` has no key | Add it (setup step 3) — launchd picks it up on its next retry |
| Startup log says `[LLM] Claude isn't available (…) — using Groq` | The reason is in the brackets — `not signed in to Claude` (never logged in, or logged out since), or `No module named 'claude_agent_sdk'` (step 2 ran with a different Python) | Run `claude` and log in, or rerun step 2, then `launchctl kickstart -k "gui/$(id -u)/com.karun.almighty"` |
| Log shows `[LLM] Claude couldn't answer (…) — Groq answers for the next 10 min` | Claude failed before saying anything — most often your plan's usage limit (`rate_limit` in the brackets), which resets every five hours, or a dropped connection | Nothing — Groq covers it, and she goes back to Claude on her own after 10 minutes (`CLAUDE_RETRY_SECS` in `config.py`) |
| Startup log says she's on `ANTHROPIC_API_KEY (billed to that key, not your subscription)` | That variable is set where she starts, and Claude Code always prefers it to your login | Remove it from `.env` (and your shell profile, for foreground runs) and restart — unless you meant to pay per use |
| She answers everything with "Sorry, I had trouble with that" | Neither model could answer — the log has an `LLM error:` line for each question, e.g. Groq's `401 … Invalid API Key` after Claude's `[LLM] Claude couldn't answer` | Fix what the log names — log in again with `claude`, or fix the key in `.env` — then `launchctl kickstart -k "gui/$(id -u)/com.karun.almighty"` |
| She says she's hit her daily limit with Groq | Claude couldn't answer either (or you run Groq only). Groq's free tier gives each model 200,000 tokens a day. When `gpt-oss-120b` runs out she answers with `gpt-oss-20b` instead — the log shows `[Daily limit] openai/gpt-oss-120b is out for … min` — and this means the 20b has run out too | Nothing to fix or restart — ask again after the wait she gave, and she goes back to the 120b on her own once its limit frees up. For higher limits, upgrade to Groq's Dev tier in the [Groq console](https://console.groq.com/settings/billing) |
| Log reaches `Ready.` but she never wakes | The last `Mic:` line names a different input than the one you're talking into — she listens on the system default, so with your AirPods in the case and the lid closed she's on the MacBook mic, which records silence. Less often, mic access is denied | Make the mic you use the default in System Settings → Sound → Input — she switches within a second and logs a new `Mic:` line. If it's already right, allow the mic in System Settings → Privacy & Security → Microphone and run `mic_test.py` |
| She hears you but doesn't wake, and you can't tell why | Usually Whisper mishearing the wake word. She keeps the start of each word and primes Whisper with the name, which took her from 13 to 15 out of 15 on recorded tries, but a mumbled or very quiet one can still come back as "all my team" | Set `export WAKE_DEBUG=1` in `.env` and restart — the log then shows a `[Wake debug]` line for every utterance she captures (length, peak level, what Whisper heard, and anything dropped as low-confidence) and a `[Mic debug]` health line every 10 s. Set it back to `0` and restart again afterwards — it writes everything said near the mic into the log |
| She wakes up when nobody called her | Something that sounds like the name — "hey, all my tea…" does it reliably, and a TV can too | Turn on `WAKE_DEBUG` (above) to see what she heard. To make her stricter, raise `WAKE_MIN_LOGPROB` in `config.py` towards `-1.0` and restart — she'll then miss more quiet or casual wakes |
| Log shows `[Mic] … — retrying every second` | Her input stream failed or stopped delivering audio, e.g. a device dropped out mid-stream | Nothing — she reopens it on her own and logs `[Mic] back on …` once audio flows. If that line never comes, check the device in System Settings → Sound |
| She keeps interrupting herself | The mic picks up her own voice from the speakers | Use headphones — see Barge-in above |
| `Bootstrap failed: 5: Input/output error` | The service is already loaded | `launchctl bootout "gui/$(id -u)/com.karun.almighty"`, then bootstrap again |
| PyAudio fails to build during `pip install` | PortAudio isn't installed | `brew install portaudio`, then rerun step 2 |
| She says "Web search unavailable" | `ddgs` isn't installed for `python3.11` | Rerun step 2 |
| She says the weather isn't available right now | `get_weather` got no answer from Open-Meteo — the Mac is offline, or the service is down or slower than its 6-second timeout. The log's `[Tool] get_weather` line shows the place she asked for | Check the Mac's internet connection and ask again — nothing to restart |

---

## License

MIT, see [LICENSE](LICENSE).
