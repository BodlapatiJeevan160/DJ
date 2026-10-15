import time
import subprocess
import os
import ctypes
from core.context_monitor import monitor
from core.notifier import manager
from core.audio import engine as audio_engine

print("--- Real Windows PCA Verification (NO MOCKING) ---")
audio_engine.start()
audio_engine.toggle(True)

# 1. Add rule
monitor.add_rule("notepad", "Proactive Notification for Notepad")
print("Rule added. Starting monitor...")
monitor.start()

time.sleep(2) # Wait for start

# Hack to intercept manager.notify and audio_engine.speak just to count them,
# but we STILL CALL the original function so the real OS behavior happens.
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

# 2. Launch Notepad
print("\nLaunching Notepad...")
proc = subprocess.Popen(["notepad.exe"])
time.sleep(2)

# Find Notepad window and force it to foreground
print("Attempting to force Notepad to foreground via ctypes...")
user32 = ctypes.windll.user32
def enum_windows_proc(hwnd, lParam):
    length = user32.GetWindowTextLengthW(hwnd)
    if length > 0:
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if "notepad" in buf.value.lower():
            print(f"Found Notepad HWND: {hwnd} ('{buf.value}')")
            user32.SetForegroundWindow(hwnd)
            return 0 # Stop enumerating
    return 1 # Continue

EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
user32.EnumWindows(EnumWindowsProc(enum_windows_proc), 0)

time.sleep(3) # Wait for PCA loop to catch it (polls every 2s)

print(f"[DEBUG] Active window reported by ContextMonitor: '{monitor._get_active_window_title()}'")

# 3. Check counts
print("\n--- After first switch ---")
print(f"Notifications: {notify_count} (Expected: 1)")
print(f"TTS calls: {tts_count} (Expected: 0)")

# 4. Switch away (minimize or kill notepad, then open again)
print("\nKilling Notepad and waiting...")
os.system("taskkill /F /IM notepad.exe >nul 2>&1")
time.sleep(3)

print("\nLaunching Notepad again (within cooldown window)...")
proc = subprocess.Popen(["notepad.exe"])
time.sleep(2)
user32.EnumWindows(EnumWindowsProc(enum_windows_proc), 0)
time.sleep(3)

print(f"[DEBUG] Active window reported by ContextMonitor: '{monitor._get_active_window_title()}'")

print("\n--- After second switch (spam check) ---")
print(f"Notifications: {notify_count} (Expected: 1)")
print(f"TTS calls: {tts_count} (Expected: 0)")

os.system("taskkill /F /IM notepad.exe >nul 2>&1")
monitor.stop()
audio_engine.stop()

if notify_count == 1 and tts_count == 0:
    print("\nSUCCESS: Verification passed!")
else:
    print("\nFAILED: Verification failed.")

# Cleanup
if os.path.exists(monitor.STORAGE_FILE):
    os.remove(monitor.STORAGE_FILE)
