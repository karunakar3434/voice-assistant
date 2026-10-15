#!/usr/bin/env python3
"""Mic tester — lists input devices and shows a live level meter.
Run:  /opt/homebrew/bin/python3.11 ~/voice-assistant/mic_test.py
      (optionally pass a device index:  ... mic_test.py 2)
Stop with Ctrl+C.
"""
import sys, time, numpy as np, pyaudio

RATE, CHUNK = 16000, 512
pa = pyaudio.PyAudio()

print("\n=== Input devices ===")
default_in = pa.get_default_input_device_info()["index"]
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        tag = "  <- DEFAULT" if i == default_in else ""
        print(f"  [{i}] {info['name']}{tag}")

dev = int(sys.argv[1]) if len(sys.argv) > 1 else default_in
print(f"\nListening on device [{dev}]. Speak now — bar should move. Ctrl+C to stop.\n")

stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                 input=True, input_device_index=dev, frames_per_buffer=CHUNK)
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms = float(np.sqrt(np.mean(np.frombuffer(data, np.int16).astype(np.float32) ** 2)))
        bar = "#" * min(60, int(rms / 50))
        print(f"\rlevel {rms:6.0f} |{bar:<60}|", end="", flush=True)
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\n\nDone.")
finally:
    stream.stop_stream(); stream.close(); pa.terminate()
