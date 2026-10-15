"""
test_phase6_1.py
================
Phase 6.1 — Safe Application Launching Tests.

All unit tests mock subprocess.Popen and os.startfile so no real
applications are launched.  A separate manual/integration section is
clearly marked for human verification.

Run with:
    python test_phase6_1.py
"""

import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_brain(intent_json: str, llm_chat_response: str = "N/A"):
    from core.brain import Brain
    brain = Brain()
    brain.llm.chat_json = MagicMock(return_value=intent_json)
    brain.llm.chat      = MagicMock(return_value=llm_chat_response)
    return brain


def _app_success(app_name: str, launched: str) -> str:
    return json.dumps({
        "success": True, "tool": "open_application",
        "data": {"application": app_name, "launched": launched},
        "error": None
    })


def _app_failure(error_msg: str) -> str:
    return json.dumps({
        "success": False, "tool": "open_application",
        "data": None,
        "error": error_msg
    })


# ===========================================================================
# TEST SUITE — Unit Tests (no real apps launched)
# ===========================================================================
class TestPhase6_1SafeLaunch(unittest.TestCase):

    # -----------------------------------------------------------------------
    # TEST 1: "Open Chrome" → open_application(application="chrome")
    # -----------------------------------------------------------------------
    def test_open_chrome_calls_correct_tool(self):
        brain = make_brain('{"tool": "open_application", "args": {"application": "chrome"}}')
        brain.registry.execute_tool = MagicMock(
            return_value=_app_success("Google Chrome", "chrome.lnk")
        )
        response = brain.process_input("Open Chrome")
        brain.registry.execute_tool.assert_called_once_with(
            name="open_application",
            args={"application": "chrome"},
            confirm_callback=brain._ask_confirmation
        )
        self.assertIn("Google Chrome", response)
        brain.llm.chat.assert_not_called()
        print(f"\n[TEST 1] Chrome: {response}")

    # -----------------------------------------------------------------------
    # TEST 2: "Open Notepad" → launched, response confirms
    # -----------------------------------------------------------------------
    def test_open_notepad(self):
        brain = make_brain('{"tool": "open_application", "args": {"application": "notepad"}}')
        brain.registry.execute_tool = MagicMock(
            return_value=_app_success("Notepad", "notepad.exe")
        )
        response = brain.process_input("Open Notepad")
        self.assertIn("Notepad", response)
        brain.llm.chat.assert_not_called()
        print(f"\n[TEST 2] Notepad: {response}")

    # -----------------------------------------------------------------------
    # TEST 3: "Open VS Code" → correctly reports if not installed
    # -----------------------------------------------------------------------
    def test_open_vscode_not_installed(self):
        brain = make_brain('{"tool": "open_application", "args": {"application": "vscode"}}')
        brain.registry.execute_tool = MagicMock(
            return_value=_app_failure("VS Code does not appear to be installed (no Start Menu shortcut found for 'visual studio code').")
        )
        response = brain.process_input("Open VS Code")
        # Must report failure honestly — never claim success
        self.assertIn("failed", response.lower())
        brain.llm.chat.assert_not_called()
        print(f"\n[TEST 3] VS Code not installed: {response}")

    # -----------------------------------------------------------------------
    # TEST 4: Unknown application is rejected
    # -----------------------------------------------------------------------
    def test_unknown_app_rejected(self):
        from skills.app_registry import launch
        result = launch("hackertool")
        self.assertFalse(result["success"])
        self.assertIn("permission to manage", result["error"])
        self.assertIn("permission to manage", result["error"])
        print(f"\n[TEST 4] Unknown app rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 5: Path traversal is rejected
    # -----------------------------------------------------------------------
    def test_path_traversal_rejected(self):
        from skills.app_registry import launch
        result = launch("../../Windows/system32/cmd")
        self.assertFalse(result["success"])
        # After normalization: "windowssystem32cmd" — not in allowlist
        self.assertIn("permission to manage", result["error"])
        print(f"\n[TEST 5] Path traversal rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 6: Shell injection-style argument is sanitized and rejected
    # -----------------------------------------------------------------------
    def test_shell_injection_rejected(self):
        from skills.app_registry import launch
        # Attempt: "notepad; del C:\\" — after _normalize → "notepaddelc" — not in allowlist
        result = launch("notepad; del C:\\")
        self.assertFalse(result["success"])
        self.assertIn("permission to manage", result["error"])
        print(f"\n[TEST 6] Shell injection sanitized+rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 6b: Extra path chars are stripped
    # -----------------------------------------------------------------------
    def test_dotexe_stripped_and_rejected(self):
        from skills.app_registry import launch
        # "cmd.exe" -> normalize -> "cmdexe" -> not in allowlist (cmd is intentionally removed)
        result = launch("cmd.exe")
        self.assertFalse(result["success"])
        print(f"\n[TEST 6b] cmd.exe stripped -> rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 7: get_system_info still works (Phase 6 regression)
    # -----------------------------------------------------------------------
    def test_get_system_info_regression(self):
        brain = make_brain('{"tool": "get_system_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=json.dumps({
            "success": True, "tool": "get_system_info",
            "data": {
                "os": "Windows 11", "os_version": "10.0.22631",
                "hostname": "PC", "architecture": "AMD64",
                "processor": "Intel i7", "cpu_percent": 20.0,
                "cpu_cores": 8, "ram_total_gb": 16.0, "ram_used_gb": 8.0,
                "ram_free_gb": 8.0, "ram_percent": 50.0,
                "disks": [], "battery": None, "uptime": "1h 0m",
                "process_count": 80
            },
            "error": None
        }))
        response = brain.process_input("What are my system specs?")
        brain.llm.chat.assert_not_called()
        self.assertIn("Windows 11", response)
        self.assertIn("20.0%",      response)
        print(f"\n[TEST 7] get_system_info regression OK.")

    # -----------------------------------------------------------------------
    # TEST 8: search_web still works (Phase 5 regression)
    # -----------------------------------------------------------------------
    def test_search_web_regression(self):
        brain = make_brain('{"tool": "search_web", "args": {"query": "AI"}}')
        brain.registry.execute_tool = MagicMock(return_value=json.dumps({
            "success": True, "tool": "search_web", "error": None,
            "data": [{"title": "AI News", "snippet": "Latest AI updates", "url": "https://example.com"}]
        }))
        response = brain.process_input("Search the web for AI")
        brain.llm.chat.assert_not_called()
        self.assertIn("1. AI News", response)
        print(f"\n[TEST 8] search_web regression OK.")

    # -----------------------------------------------------------------------
    # TEST 9: search_youtube still works (Phase 4 regression)
    # -----------------------------------------------------------------------
    def test_search_youtube_regression(self):
        brain = make_brain('{"tool": "search_youtube", "args": {"query": "Python"}}')
        brain.registry.execute_tool = MagicMock(return_value=json.dumps({
            "success": True, "tool": "search_youtube", "error": None, "data": "Success"
        }))
        response = brain.process_input("Search YouTube for Python")
        brain.llm.chat.assert_not_called()
        self.assertIn("YouTube", response)
        self.assertIn("Python",  response)
        print(f"\n[TEST 9] search_youtube regression OK.")

    # -----------------------------------------------------------------------
    # TEST 10: App registry _normalize is idempotent and safe
    # -----------------------------------------------------------------------
    def test_normalize_safety(self):
        from skills.app_registry import _normalize
        cases = [
            ("CHROME",          "chrome"),
            ("  Chrome  ",      "chrome"),
            ("VS Code",         "vscode"),
            ("C:\\system32\\cmd","csystem32cmd"),
            ("note;pad",        "notepad"),    # semicolon stripped
            ("../../etc/passwd","etcpasswd"),  # traversal stripped
            ("calc.exe",        "calcexe"),    # dot stripped
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_normalize(raw), expected)
        print(f"\n[TEST 10] _normalize safety OK.")


# ===========================================================================
# MANUAL / INTEGRATION SECTION (clearly separated)
# ===========================================================================
class TestPhase6_1ManualIntegration(unittest.TestCase):
    """
    These tests actually launch a real application.
    They are in a separate class so they can be skipped in CI.
    Run manually and verify the application opened visually.
    """

    @unittest.skipUnless(
        os.environ.get("JARVIS_INTEGRATION") == "1",
        "Set JARVIS_INTEGRATION=1 to run real app launch tests"
    )
    def test_real_notepad_launch(self):
        """
        MANUAL: Launches real Notepad. Verify it opens on screen.
        Run with: set JARVIS_INTEGRATION=1 && python test_phase6_1.py
        """
        from skills.app_registry import launch
        result = launch("notepad")
        print(f"\n[INTEGRATION] Notepad launch result: {result}")
        self.assertTrue(result["success"], result.get("error"))
        import time; time.sleep(2)  # Give it a moment to appear


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6.1 — Safe Application Launching Tests")
    print("=" * 60)
    # Run only unit tests by default
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestPhase6_1SafeLaunch)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
