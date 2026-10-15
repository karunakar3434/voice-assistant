"""
Everything touching the microphone and speakers: the continuous background
mic reader (which follows the system default input), voice-activity-based
utterance capture, Whisper transcription, and TTS playback with (raw-energy)
barge-in detection.
"""

import os
import time
import wave
import ctypes
import ctypes.util
import tempfile
import threading
import subprocess
import numpy as np
import pyaudio
from collections import deque
from faster_whisper import WhisperModel

from config import (
    CHUNK, SAMPLE_RATE, CHANNELS, FORMAT, ENERGY_THRESHOLD, SILENCE_CHUNKS, PREROLL_CHUNKS,
    MAX_RECORD_SECS, MIC_CHECK_SECS, MIC_STALL_SECS, TTS_VOICE,
    BARGE_THRESHOLD, BARGE_CHUNKS, BARGE_WINDOW, WAKE_DEBUG,
)


def rms(data: bytes) -> float:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0


def peak_rms(pcm: bytes) -> float:
    """Loudest CHUNK-sized slice of `pcm` — the same measure the capture threshold uses."""
    step = CHUNK * 2
    return max((rms(pcm[i:i + step]) for i in range(0, len(pcm), step)), default=0.0)


# ── Default input device (CoreAudio) ──────────────────────────────────────────
#
# PortAudio reads the device list once, when PyAudio() initialises, and never
# again — so it can't tell us the default input has changed. CoreAudio can,
# with one cheap property read, so MicBuffer polls that and only re-initialises
# PortAudio when the answer changes.

class _PropertyAddress(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32),
                ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32)]


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


_CA = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))
_CA.AudioObjectGetPropertyData.argtypes = [
    ctypes.c_uint32, ctypes.POINTER(_PropertyAddress), ctypes.c_uint32,
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
]
_CA.AudioObjectGetPropertyData.restype = ctypes.c_int32
_CA.AudioObjectSetPropertyData.argtypes = [
    ctypes.c_uint32, ctypes.POINTER(_PropertyAddress), ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
]
_CA.AudioObjectSetPropertyData.restype = ctypes.c_int32

_SYSTEM_OBJECT = 1   # kAudioObjectSystemObject
_DEFAULT_INPUT = _PropertyAddress(_fourcc("dIn "), _fourcc("glob"), 0)

# There's no CFRunLoop on our main thread, and without one the HAL can miss
# its own change notifications and hand back a stale default. Setting
# kAudioHardwarePropertyRunLoop to NULL tells it to run them on its own thread.
_no_runloop = ctypes.c_void_p(None)
_CA.AudioObjectSetPropertyData(
    _SYSTEM_OBJECT, ctypes.byref(_PropertyAddress(_fourcc("rnlp"), _fourcc("glob"), 0)),
    0, None, ctypes.sizeof(_no_runloop), ctypes.byref(_no_runloop),
)


def default_input_id() -> int | None:
    """CoreAudio's ID for the current default input device, or None on error."""
    device = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(device))
    err = _CA.AudioObjectGetPropertyData(
        _SYSTEM_OBJECT, ctypes.byref(_DEFAULT_INPUT), 0, None,
        ctypes.byref(size), ctypes.byref(device),
    )
    return device.value if err == 0 else None


# ── Continuous mic buffer ─────────────────────────────────────────────────────

class MicBuffer:
    """Background thread reads mic continuously — no dead zone during API calls.

    It also follows the system default input. Bound once at startup, she went
    deaf whenever the default moved on — a restart while the AirPods were in
    the case left her on the MacBook mic, which records silence with the lid
    closed, and nothing in the log said so (found Oct 2026, after a month with
    no wake logged). Now the reader checks the default every MIC_CHECK_SECS and
    reopens the stream when it changes, logging a fresh `Mic:` line. Only the
    reader thread touches the stream after startup, so no locking is needed
    around the reopen.
    """

    def __init__(self, maxchunks: int = 400):
        self._buf       = deque(maxlen=maxchunks)
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._pa        = None
        self._stream    = None
        self._device    = None
        self._name      = None
        self._announced = None
        self._open()   # here, not in the thread, so a startup failure still raises
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self):
        self._device = default_input_id()
        self._pa = pyaudio.PyAudio()
        self._name = self._pa.get_default_input_device_info()["name"]
        self._stream = self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=CHUNK,
        )
        # Only announce a real change — a reopen on the same device after an
        # error would otherwise repeat this line every few seconds.
        if (self._device, self._name) != self._announced:
            print(f"Mic: {self._name}")
            self._announced = (self._device, self._name)

    def _close(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _reader(self):
        next_check = time.monotonic() + MIC_CHECK_SECS
        last_audio = time.monotonic()
        last_error = None
        debug_start, debug_chunks, debug_peak = time.monotonic(), 0, 0.0
        while not self._stop.is_set():
            try:
                if self._stream is None:
                    self._open()
                    last_audio = time.monotonic()
                # Only read what's already buffered. PortAudio's blocking read
                # waits forever if the device stops delivering, and the device
                # check below would never run again.
                if self._stream.get_read_available() >= CHUNK:
                    data = self._stream.read(CHUNK, exception_on_overflow=False)
                    with self._lock:
                        self._buf.append(data)
                    last_audio = time.monotonic()
                    if last_error is not None:
                        print(f"[Mic] back on {self._name}")
                        last_error = None
                    if WAKE_DEBUG:
                        debug_chunks += 1
                        debug_peak = max(debug_peak, rms(data))
                else:
                    time.sleep(0.01)
                now = time.monotonic()
                if WAKE_DEBUG and now - debug_start >= 10:
                    print(f"[Mic debug] {self._name}: "
                          f"{debug_chunks * CHUNK / (now - debug_start):.0f} frames/s, "
                          f"peak RMS {debug_peak:.0f}")
                    debug_start, debug_chunks, debug_peak = now, 0, 0.0
                if now - last_audio > MIC_STALL_SECS:
                    raise OSError(f"no audio from {self._name} for {MIC_STALL_SECS:.0f}s")
                if now >= next_check:
                    next_check = now + MIC_CHECK_SECS
                    if default_input_id() != self._device:
                        self._close()   # reopened on the new default next pass
            except Exception as exc:
                # This used to `break`, which killed the reader silently and
                # left capture_utterance() waiting forever on an empty buffer.
                # A just-connected device often isn't ready for a second
                # (AirPods switching to their mic profile), so retry instead —
                # logging once per distinct error so a dead mic can't flood
                # the log.
                if str(exc) != last_error:
                    print(f"[Mic] {exc} — retrying every second")
                    last_error = str(exc)
                self._close()
                time.sleep(1)
        self._close()

    def pop(self):
        with self._lock:
            return self._buf.popleft() if self._buf else None

    def unread(self, chunks):
        """Put already-popped chunks back at the front, oldest first, so the
        next capture_utterance() sees them again."""
        with self._lock:
            self._buf.extendleft(reversed(chunks))

    def drain(self, secs: float = 0.25):
        time.sleep(secs)
        with self._lock:
            self._buf.clear()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


def capture_utterance(mic: MicBuffer, max_secs: float = MAX_RECORD_SECS,
                      max_speech_secs: float = MAX_RECORD_SECS) -> bytes:
    """Wait up to `max_secs` for speech to start, then record until
    SILENCE_CHUNKS of quiet or `max_speech_secs` of audio. Returns raw PCM
    bytes, or b"" if nobody spoke in time.

    The two limits used to be one window counted from the call: think for 8 s
    after she finished, and only the first 2 s of your sentence got through,
    with the rest arriving as a second, separate request.

    The quiet chunks just before speech starts are kept too (PREROLL_CHUNKS).
    The soft start of a word — the "h" of "hey" — sits under ENERGY_THRESHOLD,
    so without them Whisper only gets "…ey almighty" and hears things like
    "okay, all my team".
    """
    preroll       = deque(maxlen=PREROLL_CHUNKS)
    speech_buf    = []
    silent_count  = 0
    speech_started = False
    max_wait      = int(max_secs * SAMPLE_RATE / CHUNK)
    max_speech    = int(max_speech_secs * SAMPLE_RATE / CHUNK)
    waited        = 0

    while True:
        chunk = mic.pop()
        if chunk is None:
            time.sleep(0.005)
            continue
        energy = rms(chunk)

        if energy > ENERGY_THRESHOLD:
            if not speech_started:
                speech_buf.extend(preroll)
            speech_started = True
            speech_buf.append(chunk)
            silent_count = 0
        elif speech_started:
            speech_buf.append(chunk)
            silent_count += 1
            if silent_count >= SILENCE_CHUNKS:
                break
        else:
            preroll.append(chunk)
            waited += 1
            if waited >= max_wait:
                break

        if len(speech_buf) >= max_speech:
            break

    return b"".join(speech_buf)


# ── Whisper transcription ─────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes, model: WhisperModel,
               min_avg_logprob: float | None = None, **options) -> str:
    """Whisper's text for `audio_bytes`, lowercased. `options` go straight to
    model.transcribe() and override the defaults below. Segments Whisper is
    less sure of than `min_avg_logprob` are dropped — that's how the wake check
    throws away a prompt echoed back from noise (see WAKE_MIN_LOGPROB)."""
    if not audio_bytes:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    # Writing and transcribing both happen inside this try, so a failure in
    # either one (e.g. malformed audio_bytes) still hits the finally below
    # and cleans up the temp file, instead of leaking it and crashing the
    # caller — this is treated the same as "nothing usable was said."
    try:
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        segments, _ = model.transcribe(tmp_path, **{"language": "en", "beam_size": 1, **options})
        kept = []
        for seg in segments:
            if min_avg_logprob is not None and seg.avg_logprob < min_avg_logprob:
                if WAKE_DEBUG:
                    print(f'[Wake debug] dropped "{seg.text.strip()}" '
                          f"(avg_logprob {seg.avg_logprob:.2f} < {min_avg_logprob})")
                continue
            kept.append(seg.text)
        return " ".join(kept).strip().lower()
    except Exception as exc:
        print(f"  [Transcribe] error: {exc}")
        return ""
    finally:
        os.unlink(tmp_path)


# ── TTS (with barge-in) ───────────────────────────────────────────────────────
#
# Real acoustic echo cancellation was attempted here and reverted: it
# measurably failed on this setup (a chunk-by-chunk trace showed the
# echo-cancelled signal tracking the raw signal almost 1:1 with no real
# convergence, most likely because pacing playback via blocking
# out_stream.write() doesn't keep the reference and mic-capture threads
# locked to real time closely enough for the adaptive filter to track).
# That's the kind of bug that needs to be heard to debug properly, which
# isn't practical to iterate on blind. Back to plain raw-mic-RMS — use
# headphones/AirPods to sidestep the false-interrupt problem instead.

def speak(text: str, mic: MicBuffer = None, interruptible: bool = False) -> bool:
    """Speak `text`. If interruptible, stop the moment the user starts talking.
    Returns True if the user interrupted, else False."""
    print(f"  Assistant: {text}")
    # `--` ends say's options: a reply starting with "-" was being parsed as
    # one, and "-o <path>" made say write an audio file wherever it pointed.
    cmd = ["say", "-v", TTS_VOICE, "--", text]

    if not interruptible or mic is None:
        subprocess.run(cmd, check=False)
        if mic is not None:
            mic.drain(0.25)
        return False

    mic.drain(0.15)  # clear residual audio just before speaking
    proc = subprocess.Popen(cmd)

    window = deque(maxlen=BARGE_WINDOW)   # which of the latest chunks were loud
    interrupted = False
    peak = 0.0
    # The chunks that prove an interruption are the start of what you said —
    # handed back to the mic afterwards so the next capture hears them, rather
    # than losing your first half-second.
    recent = deque(maxlen=BARGE_WINDOW + PREROLL_CHUNKS)
    while proc.poll() is None:
        chunk = mic.pop()
        if chunk is None:
            time.sleep(0.005)
            continue
        recent.append(chunk)
        energy = rms(chunk)
        peak = max(peak, energy)
        window.append(energy > BARGE_THRESHOLD)
        if sum(window) >= BARGE_CHUNKS:     # mostly-loud stretch → user is interrupting
            proc.terminate()
            interrupted = True
            break

    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()

    print(f"  [Barge-in] peak mic RMS during playback: {peak:.0f} "
          f"(threshold {BARGE_THRESHOLD}){' — INTERRUPTED' if interrupted else ''}")

    if interrupted:
        mic.unread(list(recent))
    else:
        mic.drain(0.2)
    return interrupted
