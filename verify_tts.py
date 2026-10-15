import time
import logging
from skills.datetime_ops import set_reminder
from core.audio import engine as audio_engine

# Start the audio engine thread
audio_engine.start()

# Enable speech
audio_engine.toggle(True)

print("Setting reminder for 2 seconds with TTS enabled...")
set_reminder("Drink water", 2)

print("Waiting 3 seconds...")
time.sleep(3)

print("Done. Check logs for AudioEngine output.")
audio_engine.stop()
