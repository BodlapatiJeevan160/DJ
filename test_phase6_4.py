"""
test_phase6_4.py
================
Phase 6.4 — Hardened Application Management Tests.

Tests the polished open/close/status flows, already-running checks,
and strict permission rejection strings.

Run with:
    python test_phase6_4.py
"""

import sys
import json
import unittest
from unittest.mock import MagicMock, patch

def make_brain(intent_json: str, llm_chat_response: str = "N/A"):
    from core.brain import Brain
    brain = Brain()
    brain.llm.chat_json = MagicMock(return_value=intent_json)
    brain.llm.chat      = MagicMock(return_value=llm_chat_response)
    brain._ask_confirmation = MagicMock(return_value=True)
    return brain


class MockProcess:
    def __init__(self, name):
        self.info = {'name': name}


class TestPhase6_4Hardening(unittest.TestCase):

    # -----------------------------------------------------------------------
    # TEST 1: Open an app that is already running
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    @patch('subprocess.Popen')
    @patch('skills.window_ops._get_all_windows',
           return_value=[{"hwnd": 999, "title": "New Tab - Google Chrome", "pid": 1234}])
    @patch('skills.window_ops.focus_window', return_value={"success": True})
    def test_open_already_running(self, mock_focus, mock_get_windows,
                                  mock_popen, mock_process_iter):
        mock_process_iter.return_value = [MockProcess("chrome.exe")]
        brain = make_brain('{"tool": "open_application", "args": {"application": "chrome"}}')

        response = brain.process_input("Open Chrome")

        # Popen should NOT be called — existing window was focused instead
        mock_popen.assert_not_called()
        self.assertIn("Google Chrome is already running", response)
        print(f"\n[TEST 1] Open (already running): {response}")

    # -----------------------------------------------------------------------
    # TEST 2: Open an app that is not running
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    @patch('os.startfile')
    @patch('skills.app_registry._find_start_menu_shortcut',
           return_value=r'C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Google Chrome.lnk')
    @patch('skills.window_ops._get_all_windows', side_effect=[
        [],   # pre-launch running check: no existing window
        [{"hwnd": 888, "title": "New Tab - Google Chrome", "pid": 4321}],  # post-launch verify
    ])
    @patch('time.sleep')
    def test_open_not_running(self, mock_sleep, mock_get_windows,
                              mock_shortcut, mock_startfile, mock_process_iter):
        mock_process_iter.return_value = [MockProcess("explorer.exe")]
        brain = make_brain('{"tool": "open_application", "args": {"application": "chrome"}}')

        response = brain.process_input("Open Chrome")

        # startfile SHOULD be called with the mocked shortcut path
        mock_startfile.assert_called_once()
        self.assertIn("Done! I've opened Google Chrome", response)
        print(f"\n[TEST 2] Open (not running): {response}")

    # -----------------------------------------------------------------------
    # TEST 3: Close an app (success)
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_close_running(self, mock_process_iter):
        p1 = MockProcess("notepad.exe")
        p1.terminate = MagicMock()
        mock_process_iter.return_value = [p1]
        
        brain = make_brain('{"tool": "close_application", "args": {"application": "notepad"}}')
        response = brain.process_input("Terminate Notepad")
        
        p1.terminate.assert_called_once()
        self.assertIn("Done! I've closed Notepad", response)
        print(f"\n[TEST 3] Close (running): {response}")

    # -----------------------------------------------------------------------
    # TEST 4: Close an app (not running)
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_close_not_running(self, mock_process_iter):
        mock_process_iter.return_value = []
        
        brain = make_brain('{"tool": "close_application", "args": {"application": "notepad"}}')
        response = brain.process_input("Terminate Notepad")
        
        self.assertIn("Notepad is not currently running", response)
        print(f"\n[TEST 4] Close (not running): {response}")

    # -----------------------------------------------------------------------
    # TEST 5: Status (running) - Conciseness check
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_status_running(self, mock_process_iter):
        mock_process_iter.return_value = [MockProcess("chrome.exe")]
        
        brain = make_brain('{"tool": "get_application_status", "args": {"application": "chrome"}}')
        response = brain.process_input("Is Chrome running?")
        
        self.assertEqual(response.strip(), "Google Chrome is currently running.")
        print(f"\n[TEST 5] Status (running): {response}")

    # -----------------------------------------------------------------------
    # TEST 6: Unknown application rejection (polished string)
    # -----------------------------------------------------------------------
    def test_unknown_app_rejection(self):
        brain = make_brain('{"tool": "open_application", "args": {"application": "hackertool"}}')
        response = brain.process_input("Open hackertool")
        
        self.assertEqual(response.strip(), "I don't have permission to manage that application.")
        print(f"\n[TEST 6] Unknown app rejection: {response}")

    # -----------------------------------------------------------------------
    # TEST 7: Path traversal rejection (polished string)
    # -----------------------------------------------------------------------
    def test_path_traversal_rejection(self):
        brain = make_brain('{"tool": "close_application", "args": {"application": "../../Windows/system32/cmd"}}')
        response = brain.process_input("Terminate cmd")
        
        self.assertEqual(response.strip(), "I don't have permission to manage that application.")
        print(f"\n[TEST 7] Path traversal rejection: {response}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6.4 — Hardened Application Management Tests")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase6_4Hardening)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
