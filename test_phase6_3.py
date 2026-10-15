"""
test_phase6_3.py
================
Phase 6.3 — Secure Application Status Tests.

All unit tests mock psutil so no real processes are inspected.
Validates the allowlist logic, target process mapping, and failure states.

Run with:
    python test_phase6_3.py
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


class TestPhase6_3AppStatus(unittest.TestCase):

    # -----------------------------------------------------------------------
    # TEST 1: Check Chrome running
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_status_chrome_running(self, mock_process_iter):
        mock_process_iter.return_value = [
            MockProcess("chrome.exe"),
            MockProcess("chrome.exe"),
            MockProcess("explorer.exe")
        ]
        brain = make_brain('{"tool": "get_application_status", "args": {"application": "chrome"}}')
        response = brain.process_input("Is Chrome running?")
        self.assertIn("Google Chrome is currently running", response)
        self.assertIn("Google Chrome is currently running", response)
        print(f"\n[TEST 1] Chrome running: {response}")

    # -----------------------------------------------------------------------
    # TEST 2: Check Chrome NOT running
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_status_chrome_not_running(self, mock_process_iter):
        mock_process_iter.return_value = [MockProcess("explorer.exe")]
        brain = make_brain('{"tool": "get_application_status", "args": {"application": "chrome"}}')
        response = brain.process_input("Is Chrome running?")
        self.assertIn("Google Chrome is not currently running", response)
        print(f"\n[TEST 2] Chrome not running: {response}")

    # -----------------------------------------------------------------------
    # TEST 3: List all running applications (filters out non-allowed)
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_status_all_running(self, mock_process_iter):
        mock_process_iter.return_value = [
            MockProcess("chrome.exe"),
            MockProcess("code.exe"),
            MockProcess("svchost.exe"),  # Should be hidden
            MockProcess("unknown.exe")   # Should be hidden
        ]
        brain = make_brain('{"tool": "get_application_status", "args": {"application": "all"}}')
        response = brain.process_input("What applications are running?")
        self.assertIn("Google Chrome", response)
        self.assertIn("VS Code", response)
        self.assertNotIn("svchost", response.lower())
        self.assertNotIn("unknown", response.lower())
        print(f"\n[TEST 3] List all running (filtered): {response}")

    # -----------------------------------------------------------------------
    # TEST 4: Unknown application rejected
    # -----------------------------------------------------------------------
    def test_status_unknown_app_rejected(self):
        from skills.app_registry import get_status
        result = get_status("hackertool")
        self.assertFalse(result["success"])
        self.assertIn("permission to manage", result["error"])
        print(f"\n[TEST 4] Unknown app rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 5: Path traversal rejected
    # -----------------------------------------------------------------------
    def test_status_path_traversal_rejected(self):
        from skills.app_registry import get_status
        result = get_status("../../Windows/system32/cmd")
        self.assertFalse(result["success"])
        self.assertIn("permission to manage", result["error"])
        print(f"\n[TEST 5] Path traversal rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 6: Permission error handled gracefully
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_status_permission_error(self, mock_process_iter):
        import psutil
        def mock_iter(*args, **kwargs):
            raise psutil.AccessDenied(pid=123)
        mock_process_iter.side_effect = mock_iter
        
        from skills.app_registry import get_status
        result = get_status("all")
        self.assertFalse(result["success"])
        self.assertIn("Failed to list running", result["error"])
        print(f"\n[TEST 6] Permission error handled gracefully: {result['error'][:80]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6.3 — Secure Application Status Tests")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase6_3AppStatus)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
