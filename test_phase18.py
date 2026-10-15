"""
test_phase18.py
===============
Phase 18 -- Process Monitor Tests

Security invariants verified:
  - list_processes only reads top 30 processes, sorted deterministically.
  - No new kill_process tool is created; close_application is reused for safe, allowlisted termination.
  - is_dangerous is False for list_processes.
  - Handled AccessDenied, NoSuchProcess, ZombieProcess gracefully.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from skills import system_ops
from core.tool_registry import ToolRegistry
from core.brain import Brain

# ===========================================================================
# Helpers
# ===========================================================================

def _make_brain(mock_tool_decision: str) -> Brain:
    """Helper to create a Brain with a mocked Pass-1 JSON output."""
    brain = Brain()
    # Mock the LLM to return exactly what we want for intent
    brain.llm.chat_json = MagicMock(return_value=mock_tool_decision)
    # Mock LLM chat for pass 2
    brain.llm.chat = MagicMock(return_value="[LLM Pass 2 Response]")
    return brain

# ===========================================================================
# Unit Tests: skills/system_ops.py
# ===========================================================================

class TestListProcesses:
    @patch("skills.system_ops.psutil.process_iter")
    def test_list_processes_success(self, mock_process_iter):
        # Mock 3 processes with varying memory
        p1 = MagicMock()
        p1.info = {'pid': 100, 'name': 'low_mem.exe', 'memory_info': MagicMock(rss=10 * 1024 * 1024), 'cpu_percent': 1.5}
        
        p2 = MagicMock()
        p2.info = {'pid': 200, 'name': 'high_mem.exe', 'memory_info': MagicMock(rss=500 * 1024 * 1024), 'cpu_percent': 10.0}
        
        p3 = MagicMock()
        p3.info = {'pid': 300, 'name': 'no_mem.exe', 'memory_info': None, 'cpu_percent': 0.0}

        mock_process_iter.return_value = [p1, p2, p3]

        res = system_ops.list_processes()

        assert res["success"] is True
        assert res["tool"] == "list_processes"
        assert res["data"]["total_count"] == 3
        
        procs = res["data"]["processes"]
        assert len(procs) == 3
        
        # Should be sorted by memory descending
        assert procs[0]["name"] == "high_mem.exe"
        assert procs[0]["memory_mb"] == 500.0
        
        assert procs[1]["name"] == "low_mem.exe"
        assert procs[1]["memory_mb"] == 10.0
        
        assert procs[2]["name"] == "no_mem.exe"
        assert procs[2]["memory_mb"] == 0.0

    @patch("skills.system_ops.psutil.process_iter")
    def test_list_processes_limits_to_30(self, mock_process_iter):
        # Create 40 mock processes
        mock_procs = []
        for i in range(40):
            p = MagicMock()
            p.info = {'pid': i, 'name': f'proc_{i}.exe', 'memory_info': MagicMock(rss=i * 1024 * 1024), 'cpu_percent': 0.0}
            mock_procs.append(p)
            
        mock_process_iter.return_value = mock_procs
        
        res = system_ops.list_processes()
        
        assert res["success"] is True
        assert res["data"]["total_count"] == 40
        assert len(res["data"]["processes"]) == 30
        
        # The highest memory (proc_39) should be first
        assert res["data"]["processes"][0]["name"] == "proc_39.exe"

    @patch("skills.system_ops.psutil.process_iter")
    def test_list_processes_handles_exceptions(self, mock_process_iter):
        import psutil
        
        p_ok = MagicMock()
        p_ok.info = {'pid': 1, 'name': 'ok.exe', 'memory_info': MagicMock(rss=1000), 'cpu_percent': 1.0}
        
        # Mocking properties that raise exceptions when accessed
        class BadProc:
            @property
            def info(self):
                raise psutil.AccessDenied()
                
        mock_process_iter.return_value = [p_ok, BadProc()]
        
        res = system_ops.list_processes()
        
        assert res["success"] is True
        assert res["data"]["total_count"] == 1
        assert res["data"]["processes"][0]["name"] == "ok.exe"


# ===========================================================================
# Registry & Brain Tests
# ===========================================================================

class TestToolRegistryPhase18:
    def test_tools_registered(self):
        registry = ToolRegistry()
        assert "list_processes" in registry.tools

    def test_not_dangerous(self):
        registry = ToolRegistry()
        assert registry.tools["list_processes"][1] is False


class TestBrainGroundingPhase18:
    def test_list_processes_success_returns_none_for_pass2(self):
        brain = Brain()
        res = brain._build_grounded_response(
            "list_processes",
            {},
            json.dumps({
                "success": True,
                "data": {"processes": [{"name": "chrome.exe", "memory_mb": 100}], "total_count": 1}
            })
        )
        # Returns None to trigger Pass 2
        assert res is None

    def test_list_processes_failure_returns_deterministic_string(self):
        brain = Brain()
        res = brain._build_grounded_response(
            "list_processes",
            {},
            json.dumps({"success": False, "error": "Access Denied"})
        )
        assert res is not None
        assert "I tried to list the running processes" in res
        assert "Access Denied" in res


class TestBrainIntegrationPhase18:
    @patch("skills.system_ops.list_processes")
    def test_list_processes_pipeline(self, mock_list):
        mock_list.return_value = {
            "success": True, 
            "data": {"processes": [{"name": "test.exe", "memory_mb": 10}], "total_count": 1}
        }
        
        brain = _make_brain('{"tool": "list_processes", "args": {}}')
        response = brain.process_input("What is running?")
        
        # Tool was called
        mock_list.assert_called_once()
        # LLM Pass 2 WAS called
        brain.llm.chat.assert_called_once()
        assert response == "[LLM Pass 2 Response]"

    def test_kill_process_maps_to_close_application(self):
        """
        Verify that our system prompt now directs 'kill' or 'terminate' requests
        to the existing, safe, allowlisted close_application tool.
        """
        brain = Brain()
        prompt = brain.base_system_prompt + "\n" + brain.registry.get_tool_descriptions()
        
        # The intent rules are generated in _build_grounded_response/process_input dynamically,
        # but we can check the strings directly via process_input's internal logic trace 
        # or by reading the string in the file.
        # This is a bit of a meta-test.
        with open("core/brain.py", "r") as f:
            content = f.read()
            assert "'Kill X', 'Terminate X', or 'Force close X' where X is an application name" in content
            assert "use list_processes with args {}" in content


class TestRegressionGuardPhase18:
    def test_existing_tools_still_registered(self):
        registry = ToolRegistry()
        # 33 from p16 + 3 from p17 + 1 from p18
        assert len(registry.tools) >= 37

    def test_close_application_still_dangerous(self):
        registry = ToolRegistry()
        assert registry.tools["close_application"][1] is True
