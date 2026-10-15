import time
import pyautogui
from skills.datetime_ops import set_reminder

print("Setting reminder for 3 seconds...")
set_reminder("Test Desktop Notification!", 3)

print("Waiting for notification to appear...")
time.sleep(3.5) # Wait for timer + small buffer for toast to slide in

print("Taking screenshot...")
screenshot = pyautogui.screenshot()
screenshot.save("notification_verification.png")
print("Screenshot saved to notification_verification.png")
