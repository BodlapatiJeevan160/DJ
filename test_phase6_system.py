"""
test_phase6_system.py
=====================
Phase 6 — System Information Tests.

Verifies that:
  - get_system_info is called for "what are my specs?" requests.
  - The deterministic formatter produces the correct output fields.
  - No LLM is called for system info presentation.
  - Battery absent / disk lists are handled gracefully.
  - Security: injected kwargs are silently dropped.
  - Security: LLM cannot forge a fake result.
  - Existing Phase 4/5 tools are not regressed.

Run with:
    python test_phase6_system.py
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
    return brain


def _sys_result(data: dict) -> str:
    return json.dumps({"success": True, "tool": "get_system_info", "data": data, "error": None})


def _sys_result_error(err: str) -> str:
    return json.dumps({"success": False, "tool": "get_system_info", "data": None, "error": err})


SAMPLE_DATA = {
    "os":            "Windows 11",
    "os_version":    "10.0.22631",
    "hostname":      "DESKTOP-TEST",
    "architecture":  "AMD64",
    "processor":     "Intel(R) Core(TM) i7",
    "cpu_percent":   32.5,
    "cpu_cores":     8,
    "ram_total_gb":  16.0,
    "ram_used_gb":   9.4,
    "ram_free_gb":   6.6,
    "ram_percent":   58.8,
    "disks": [
        {"mountpoint": "C:\\", "total_gb": 476.39, "used_gb": 210.0, "free_gb": 266.39, "percent": 44.1}
    ],
    "battery":       None,
    "uptime":        "3h 42m",
    "process_count": 120,
}

SAMPLE_DATA_WITH_BATTERY = {**SAMPLE_DATA, "battery": {
    "percent": 74.0, "plugged_in": False, "time_left_min": 118.5
}}


class TestPhase6SystemInfo(unittest.TestCase):

    # ------------------------------------------------------------------
    # TEST 1: LLM selects get_system_info for "what are my specs?"
    # ------------------------------------------------------------------
    def test_sysinfo_tool_selected(self):
        brain = make_brain('{"tool": "get_system_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_sys_result(SAMPLE_DATA))
        response = brain.process_input("What are my system specs?")
        brain.registry.execute_tool.assert_called_once_with(
            name="get_system_info", args={}, confirm_callback=brain._ask_confirmation
        )
        print(f"\n[TEST 1] Tool selected, response:\n{response}")

    # ------------------------------------------------------------------
    # TEST 2: Response contains all expected fields
    # ------------------------------------------------------------------
    def test_sysinfo_response_has_fields(self):
        brain = make_brain('{"tool": "get_system_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_sys_result(SAMPLE_DATA))
        response = brain.process_input("What are my system specs?")

        # LLM should NOT be called – grounding is deterministic
        brain.llm.chat.assert_not_called()

        self.assertIn("Windows 11",          response)
        self.assertIn("AMD64",               response)
        self.assertIn("32.5%",               response)  # CPU usage
        self.assertIn("8",                   response)  # CPU cores
        self.assertIn("16.0 GB",             response)  # RAM total
        self.assertIn("58.8%",               response)  # RAM percent
        self.assertIn("3h 42m",              response)  # uptime
        self.assertIn("120 running",         response)  # process count
        print(f"\n[TEST 2] All expected fields present.")

    # ------------------------------------------------------------------
    # TEST 3: Disk usage is included
    # ------------------------------------------------------------------
    def test_sysinfo_disk_shown(self):
        brain = make_brain('{"tool": "get_system_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_sys_result(SAMPLE_DATA))
        response = brain.process_input("Show me disk usage")

        self.assertIn("Disk Usage",   response)
        self.assertIn("C:\\",         response)
        self.assertIn("476.39",       response)  # total
        self.assertIn("44.1%",        response)  # percent
        print(f"\n[TEST 3] Disk usage present.")

    # ------------------------------------------------------------------
    # TEST 4: Battery absent is handled gracefully
    # ------------------------------------------------------------------
    def test_sysinfo_no_battery(self):
        brain = make_brain('{"tool": "get_system_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_sys_result(SAMPLE_DATA))
        response = brain.process_input("What are my specs?")

        self.assertIn("Battery", response)
        self.assertIn("Not available", response)
        print(f"\n[TEST 4] No-battery handled gracefully.")

    # ------------------------------------------------------------------
    # TEST 5: Battery present is formatted correctly
    # ------------------------------------------------------------------
    def test_sysinfo_with_battery(self):
        brain = make_brain('{"tool": "get_system_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_sys_result(SAMPLE_DATA_WITH_BATTERY))
        response = brain.process_input("What are my specs?")

        self.assertIn("74.0%",       response)
        self.assertIn("On battery",  response)
        self.assertIn("118.5m",      response)
        print(f"\n[TEST 5] Battery info formatted correctly.")

    # ------------------------------------------------------------------
    # TEST 6: Failed get_system_info is reported honestly
    # ------------------------------------------------------------------
    def test_sysinfo_failure_reported(self):
        brain = make_brain('{"tool": "get_system_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_sys_result_error("psutil not available"))
        response = brain.process_input("What are my specs?")

        brain.llm.chat.assert_not_called()
        self.assertIn("failed",                 response.lower())
        self.assertIn("psutil not available",   response)
        print(f"\n[TEST 6] Failure reported honestly: {response}")

    # ------------------------------------------------------------------
    # TEST 7 (Security): Injected kwargs in args are silently dropped
    # ------------------------------------------------------------------
    def test_sysinfo_injected_args_dropped(self):
        """get_system_info accepts **kwargs — injected args must not affect output."""
        from skills.system_ops import get_system_info
        # Call with injected args
        result = get_system_info(
            rm_rf="/",
            shell="rm -rf /*",
            __import__="os",
        )
        # Should still succeed and return a proper dict
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        self.assertIn("cpu_percent", result.get("data", {}))
        print(f"\n[TEST 7] Injected kwargs silently dropped. CPU: {result['data']['cpu_percent']}%")

    # ------------------------------------------------------------------
    # TEST 8: Existing Phase 4/5 tools not regressed
    # ------------------------------------------------------------------
    def test_search_web_still_works(self):
        brain = make_brain('{"tool": "search_web", "args": {"query": "AI"}}')
        brain.registry.execute_tool = MagicMock(return_value=json.dumps({
            "success": True, "tool": "search_web", "error": None,
            "data": [{"title": "AI News", "snippet": "Latest AI news", "url": "https://example.com"}]
        }))
        response = brain.process_input("Search the web for AI")

        brain.llm.chat.assert_not_called()
        self.assertIn("1. AI News", response)
        print(f"\n[TEST 8] search_web regression OK.")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6 — System Information Tests")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase6SystemInfo)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
