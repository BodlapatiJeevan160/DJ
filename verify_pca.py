import time
import subprocess
import os
import ctypes
from core.context_monitor import monitor
from core.notifier import manager
from core.audio import engine as audio_engine

print("--- Real Windows PCA Verification ---")
audio_engine.start()
audio_engine.toggle(True)

# 1. Add rule
monitor.add_rule("notepad", "Proactive Notification for Notepad")
print("Rule added. Starting monitor...")
monitor.start()

time.sleep(2) # Wait for start

# Hack to intercept manager.notify and audio_engine.speak just to count them
notify_count = 0
tts_count = 0

original_notify = manager.notify
def mocked_notify(*args, **kwargs):
    global notify_count
    notify_count += 1
    print(f"[VERIFY] Desktop Notification triggered! (Total: {notify_count})")
    return original_notify(*args, **kwargs)
manager.notify = mocked_notify

original_speak = audio_engine.speak
def mocked_speak(*args, **kwargs):
    global tts_count
    tts_count += 1
    print(f"[VERIFY] TTS triggered! (Total: {tts_count})")
    return original_speak(*args, **kwargs)
audio_engine.speak = mocked_speak

# 2. Simulate opening Notepad
print("\nSimulating Notepad opening...")
def mock_notepad():
    return "Untitled - Notepad"
monitor._get_active_window_title = mock_notepad
time.sleep(3) # Wait for PCA loop to catch it (polls every 2s)

# 3. Check counts
print("\n--- After first switch ---")
print(f"Notifications: {notify_count} (Expected: 1)")
print(f"TTS calls: {tts_count} (Expected: 0)")

# 4. Switch away (minimize or kill notepad, then open again)
print("\nSimulating Notepad closing and waiting...")
def mock_empty():
    return ""
monitor._get_active_window_title = mock_empty
time.sleep(3)

print("\nSimulating Notepad opening again (within cooldown window)...")
monitor._get_active_window_title = mock_notepad
time.sleep(3)

print("\n--- After second switch (spam check) ---")
print(f"Notifications: {notify_count} (Expected: 1)")
print(f"TTS calls: {tts_count} (Expected: 0)")

monitor.stop()
audio_engine.stop()

if notify_count == 1 and tts_count == 0:
    print("\nSUCCESS: Verification passed!")
else:
    print("\nFAILED: Verification failed.")

# Cleanup
os.remove(monitor.STORAGE_FILE)
