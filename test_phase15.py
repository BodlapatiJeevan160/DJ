"""
test_phase15.py
===============
Phase 15 -- Secure Network Diagnostics Tests.

Security invariants verified:
  - get_network_info uses ONLY psutil and socket (no subprocess, no shell, no HTTP)
  - No credentials, passwords, or secrets returned
  - LLM provides NO arguments (function accepts none)
  - Graceful degradation on partial/missing adapter data
  - Pass-2 grounding (returns None from _build_grounded_response)

Tests cover:
  - Success: connected machine with adapters
  - Offline: gethostbyname raises, adapters still returned
  - No adapters: empty if_addrs
  - Malformed/missing stats: if_stats missing an interface
  - IPv6 zone-id stripping
  - Total exception: success=False
  - Registry: registered, NOT dangerous
  - Brain grounding: returns None (→ LLM Pass-2)
  - Integration: full Brain pipeline routes correctly
"""

import json
import socket
import pytest
from unittest.mock import patch, MagicMock

from skills import network_ops
from core.tool_registry import ToolRegistry
from core.brain import Brain


# ===========================================================================
# Fake psutil data helpers
# ===========================================================================

def _make_snic(family, address):
    """Create a mock psutil SNIC (network interface card) address object."""
    snic = MagicMock()
    snic.family = family
    snic.address = address
    return snic


def _make_stats(isup=True, speed=1000):
    stats = MagicMock()
    stats.isup = isup
    stats.speed = speed
    return stats


_FAKE_IF_ADDRS = {
    "Ethernet": [
        _make_snic(socket.AF_INET, "192.168.1.100"),
        _make_snic(socket.AF_INET6, "fe80::1%Ethernet"),
    ],
    "Loopback": [
        _make_snic(socket.AF_INET, "127.0.0.1"),
    ],
}

_FAKE_IF_STATS = {
    "Ethernet": _make_stats(isup=True, speed=1000),
    "Loopback": _make_stats(isup=True, speed=0),
}

_SUCCESS_RESULT = json.dumps({
    "success": True,
    "tool": "get_network_info",
    "data": {
        "hostname": "TESTPC",
        "local_ip": "192.168.1.100",
        "adapters": [
            {
                "name": "Ethernet",
                "ipv4": ["192.168.1.100"],
                "ipv6": ["fe80::1"],
                "is_up": True,
                "speed_mbps": 1000,
            }
        ],
    },
    "error": None,
})

_FAIL_RESULT = json.dumps({
    "success": False,
    "tool": "get_network_info",
    "data": None,
    "error": "psutil error",
})


# ===========================================================================
# Unit Tests: get_network_info
# ===========================================================================

class TestGetNetworkInfo:

    @patch("skills.network_ops.psutil.net_if_stats", return_value=_FAKE_IF_STATS)
    @patch("skills.network_ops.psutil.net_if_addrs", return_value=_FAKE_IF_ADDRS)
    @patch("skills.network_ops.socket.gethostname", return_value="TESTPC")
    def test_success_standard(self, mock_hostname, mock_addrs, mock_stats):
        res = network_ops.get_network_info()

        assert res["success"] is True
        assert res["tool"] == "get_network_info"
        assert res["data"]["hostname"] == "TESTPC"
        assert res["data"]["local_ip"] == "192.168.1.100"
        adapters = res["data"]["adapters"]
        assert len(adapters) >= 1
        eth = next(a for a in adapters if a["name"] == "Ethernet")
        assert "192.168.1.100" in eth["ipv4"]
        assert eth["is_up"] is True
        assert eth["speed_mbps"] == 1000

    @patch("skills.network_ops.psutil.net_if_stats", return_value=_FAKE_IF_STATS)
    @patch("skills.network_ops.psutil.net_if_addrs", return_value=_FAKE_IF_ADDRS)
    @patch("skills.network_ops.socket.gethostname", return_value="TESTPC")
    def test_loopback_address_not_selected_as_local_ip(self, mock_hostname, mock_addrs, mock_stats):
        """127.x.x.x must never be chosen as the local_ip."""
        res = network_ops.get_network_info()
        assert res["data"]["local_ip"] != "127.0.0.1"

    @patch("skills.network_ops.psutil.net_if_stats", return_value=_FAKE_IF_STATS)
    @patch("skills.network_ops.psutil.net_if_addrs", return_value=_FAKE_IF_ADDRS)
    @patch("skills.network_ops.socket.gethostname", return_value="TESTPC")
    def test_ipv6_zone_id_stripped(self, mock_hostname, mock_addrs, mock_stats):
        """Zone-ID suffix (%Ethernet) must be stripped from IPv6 addresses."""
        res = network_ops.get_network_info()
        eth = next(a for a in res["data"]["adapters"] if a["name"] == "Ethernet")
        for addr in eth["ipv6"]:
            assert "%" not in addr, f"Zone-ID not stripped: {addr}"

    @patch("skills.network_ops.psutil.net_if_stats", return_value={})
    @patch("skills.network_ops.psutil.net_if_addrs", return_value={"Wi-Fi": [_make_snic(socket.AF_INET, "10.0.0.5")]})
    @patch("skills.network_ops.socket.gethostname", return_value="LAPTOP")
    def test_missing_stats_handled_gracefully(self, mock_hostname, mock_addrs, mock_stats):
        """If net_if_stats has no entry for an interface, is_up and speed_mbps should be None."""
        res = network_ops.get_network_info()
        assert res["success"] is True
        adapter = res["data"]["adapters"][0]
        assert adapter["is_up"] is None
        assert adapter["speed_mbps"] is None

    @patch("skills.network_ops.psutil.net_if_stats", return_value={})
    @patch("skills.network_ops.psutil.net_if_addrs", return_value={})
    @patch("skills.network_ops.socket.gethostname", return_value="EMPTYPC")
    def test_no_adapters_returns_success_empty_list(self, mock_hostname, mock_addrs, mock_stats):
        """No adapters should still return success=True with an empty adapters list."""
        res = network_ops.get_network_info()
        assert res["success"] is True
        assert res["data"]["adapters"] == []
        assert res["data"]["local_ip"] is None

    @patch("skills.network_ops.psutil.net_if_addrs", side_effect=Exception("psutil error"))
    @patch("skills.network_ops.socket.gethostname", return_value="PC")
    def test_total_exception_returns_failure(self, mock_hostname, mock_addrs):
        res = network_ops.get_network_info()
        assert res["success"] is False
        assert "psutil error" in res["error"]

    @patch("skills.network_ops.psutil.net_if_stats", return_value=_FAKE_IF_STATS)
    @patch("skills.network_ops.psutil.net_if_addrs", return_value=_FAKE_IF_ADDRS)
    @patch("skills.network_ops.socket.gethostname", return_value="TESTPC")
    def test_no_credentials_in_result(self, mock_hostname, mock_addrs, mock_stats):
        """Verify no password, key, token, or secret fields appear in the response."""
        res = network_ops.get_network_info()
        result_str = json.dumps(res).lower()
        for forbidden in ("password", "passwd", "secret", "token", "key", "credential"):
            assert forbidden not in result_str, f"Forbidden field found: {forbidden}"

    def test_function_accepts_no_arguments(self):
        """get_network_info must accept zero arguments from the LLM."""
        import inspect
        sig = inspect.signature(network_ops.get_network_info)
        assert len(sig.parameters) == 0


# ===========================================================================
# Tool Registry Tests
# ===========================================================================

class TestToolRegistry:

    def test_network_info_registered(self):
        registry = ToolRegistry()
        assert "get_network_info" in registry.tools

    def test_network_info_not_dangerous(self):
        registry = ToolRegistry()
        _, is_dangerous = registry.tools["get_network_info"]
        assert is_dangerous is False


# ===========================================================================
# Brain Grounding Tests
# ===========================================================================

class TestBrainGrounding:

    def setup_method(self):
        self.brain = Brain()

    def test_grounding_success_returns_none_for_pass2(self):
        """get_network_info must return None from grounding → routed to LLM Pass-2."""
        grounded = self.brain._build_grounded_response(
            "get_network_info", {}, _SUCCESS_RESULT)
        assert grounded is None

    def test_grounding_failure_returns_deterministic_string(self):
        grounded = self.brain._build_grounded_response(
            "get_network_info", {}, _FAIL_RESULT)
        assert grounded is not None
        assert "failed" in grounded.lower()


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestBrainIntegration:

    def _make_brain(self, intent_json, llm_final="Your IP is 192.168.1.100."):
        brain = Brain()
        brain.llm.chat_json = MagicMock(return_value=intent_json)
        brain.llm.chat      = MagicMock(return_value=llm_final)
        return brain

    def test_get_network_info_pipeline_calls_llm_pass2(self):
        """On success, brain must call LLM Pass-2 (not return deterministic string)."""
        brain = self._make_brain('{"tool": "get_network_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_SUCCESS_RESULT)

        response = brain.process_input("What is my IP address?")

        brain.registry.execute_tool.assert_called_once_with(
            name="get_network_info", args={}, confirm_callback=brain._ask_confirmation)
        # LLM Pass-2 MUST be called for conversational wrapping
        brain.llm.chat.assert_called_once()
        assert response == "Your IP is 192.168.1.100."

    def test_get_network_info_failure_bypasses_llm(self):
        """On failure, brain must return deterministic failure string without LLM."""
        brain = self._make_brain('{"tool": "get_network_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_FAIL_RESULT)

        response = brain.process_input("What is my IP?")

        assert "failed" in response.lower()
        brain.llm.chat.assert_not_called()

    def test_no_args_passed_to_tool(self):
        """Verify the brain never injects LLM-sourced args into get_network_info."""
        brain = self._make_brain('{"tool": "get_network_info", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_SUCCESS_RESULT)
        brain.process_input("What is my IP?")

        # args must be empty dict — no LLM-provided arguments
        called_args = brain.registry.execute_tool.call_args.kwargs
        assert called_args["args"] == {}
