"""
test_phase19.py
===============
Phase 19 -- Persistent Memory Tests

Verifies:
  - MemoryStore constraints (limits, sizes, bad data)
  - Tool interfaces (remember, recall, forget, list_memory)
  - Brain injection (security boundary for injected memory)
  - ToolRegistry configurations
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from core.memory import MemoryStore, remember, recall, forget, list_memory
from core.tool_registry import ToolRegistry
from core.brain import Brain

# ===========================================================================
# Helpers & Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def isolated_memory(tmp_path):
    """
    Forces MemoryStore to use a temporary directory for all tests,
    preventing state leakage or corrupting real user memory.
    """
    old_dir = MemoryStore.STORAGE_DIR
    old_file = MemoryStore.STORAGE_FILE
    
    test_dir = str(tmp_path / ".jarvis_test")
    MemoryStore.STORAGE_DIR = test_dir
    MemoryStore.STORAGE_FILE = os.path.join(test_dir, "memory.json")
    
    yield
    
    MemoryStore.STORAGE_DIR = old_dir
    MemoryStore.STORAGE_FILE = old_file

def _make_brain(mock_tool_decision: str) -> Brain:
    """Helper to create a Brain with a mocked Pass-1 JSON output."""
    brain = Brain()
    brain.llm.chat_json = MagicMock(return_value=mock_tool_decision)
    brain.llm.chat = MagicMock(return_value="[LLM Pass 2 Response]")
    return brain

# ===========================================================================
# Unit Tests: core/memory.py (MemoryStore)
# ===========================================================================

class TestMemoryStoreValidation:
    def test_valid_key_value(self):
        assert MemoryStore._validate_key("test_key") == "test_key"
        assert MemoryStore._validate_value("test_val") == "test_val"

    def test_null_bytes_rejected(self):
        with pytest.raises(ValueError):
            MemoryStore._validate_key("bad\x00key")
        with pytest.raises(ValueError):
            MemoryStore._validate_value("bad\x00val")

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            MemoryStore._validate_key("   ")
        with pytest.raises(ValueError):
            MemoryStore._validate_value("")

    def test_length_limits(self):
        MemoryStore._validate_key("a" * 1000) # Should pass
        with pytest.raises(ValueError):
            MemoryStore._validate_key("a" * 1001)
            
        MemoryStore._validate_value("a" * 2000) # Should pass
        with pytest.raises(ValueError):
            MemoryStore._validate_value("a" * 2001)

    def test_types(self):
        with pytest.raises(ValueError):
            MemoryStore._validate_key(123)
        with pytest.raises(ValueError):
            MemoryStore._validate_value(["not", "a", "string"])


class TestMemoryStoreIO:
    def test_load_nonexistent_returns_empty(self):
        assert MemoryStore.load_memory() == {}

    def test_save_and_load(self):
        assert MemoryStore._save_memory({"test": "data"}) is True
        data = MemoryStore.load_memory()
        assert data == {"test": "data"}
        
    def test_load_ignores_oversized_file(self):
        os.makedirs(MemoryStore.STORAGE_DIR, exist_ok=True)
        with open(MemoryStore.STORAGE_FILE, 'w', encoding='utf-8') as f:
            f.write("a" * (MemoryStore.MAX_FILE_SIZE_BYTES + 10))
            
        # Should return empty dict instead of trying to parse huge file
        assert MemoryStore.load_memory() == {}

    def test_load_ignores_corrupt_json(self):
        os.makedirs(MemoryStore.STORAGE_DIR, exist_ok=True)
        with open(MemoryStore.STORAGE_FILE, 'w', encoding='utf-8') as f:
            f.write("{bad json]")
            
        assert MemoryStore.load_memory() == {}

    def test_load_enforces_limits_on_corrupt_entries(self):
        MemoryStore._save_memory({"bad\x00key": "val", "good": "data"})
        data = MemoryStore.load_memory()
        # The bad key should have been skipped during load validation
        assert "bad\x00key" not in data
        assert "good" in data

    def test_save_atomic_failure_does_not_corrupt(self):
        # Write valid file
        MemoryStore._save_memory({"valid": "data"})
        
        # Force a failure during save (e.g. mock os.replace)
        with patch('os.replace', side_effect=Exception("Disk full")):
            assert MemoryStore._save_memory({"new": "data"}) is False
            
        # Original file should be intact
        data = MemoryStore.load_memory()
        assert data == {"valid": "data"}


# ===========================================================================
# Unit Tests: core/memory.py (Tool Interfaces)
# ===========================================================================

class TestMemoryTools:
    def test_remember_and_recall(self):
        res = remember("name", "JARVIS")
        assert res["success"] is True
        
        res = recall("name")
        assert res["success"] is True
        assert res["data"]["value"] == "JARVIS"
        
    def test_remember_update(self):
        remember("color", "blue")
        remember("color", "red")
        assert recall("color")["data"]["value"] == "red"
        
    def test_remember_capacity_limit(self):
        # Fill memory
        data = {f"k{i}": "v" for i in range(100)}
        MemoryStore._save_memory(data)
        
        # Try to add 101st
        res = remember("new_key", "value")
        assert res["success"] is False
        assert "Memory is full" in res["error"]
        
        # Updating an existing key should still work
        res = remember("k0", "new_val")
        assert res["success"] is True

    def test_recall_not_found(self):
        res = recall("nope")
        assert res["success"] is False
        assert "not found" in res["error"]

    def test_forget(self):
        remember("temp", "data")
        assert forget("temp")["success"] is True
        assert recall("temp")["success"] is False
        
    def test_forget_not_found(self):
        assert forget("nope")["success"] is False
        
    def test_list_memory(self):
        remember("a", "1")
        remember("b", "2")
        res = list_memory()
        assert res["success"] is True
        assert res["data"]["count"] == 2
        # Order may not be guaranteed, so check presence
        keys = [x["key"] for x in res["data"]["entries"]]
        assert "a" in keys
        assert "b" in keys


# ===========================================================================
# Registry & Brain Tests
# ===========================================================================

class TestToolRegistryPhase19:
    def test_tools_registered(self):
        registry = ToolRegistry()
        assert "remember" in registry.tools
        assert "recall" in registry.tools
        assert "forget" in registry.tools
        assert "list_memory" in registry.tools

    def test_not_dangerous(self):
        registry = ToolRegistry()
        assert registry.tools["remember"][1] is False
        assert registry.tools["recall"][1] is False
        assert registry.tools["forget"][1] is False
        assert registry.tools["list_memory"][1] is False


class TestBrainMemoryIntegration:
    def test_get_memory_block_empty(self):
        brain = Brain()
        assert brain._get_memory_block() == ""
        
    def test_get_memory_block_populated(self):
        remember("user", "Alice")
        brain = Brain()
        block = brain._get_memory_block()
        assert "USER MEMORY" in block
        assert "UNTRUSTED DATA" in block
        assert "- user: Alice" in block
        
    def test_grounding_remember_forget_bypasses_pass2(self):
        brain = Brain()
        res_r = brain._build_grounded_response("remember", {"key": "x"}, json.dumps({"success": True}))
        assert "Done!" in res_r
        
        res_f = brain._build_grounded_response("forget", {"key": "x"}, json.dumps({"success": True}))
        assert "Done!" in res_f

    def test_grounding_recall_list_uses_pass2(self):
        brain = Brain()
        res_rec = brain._build_grounded_response("recall", {"key": "x"}, json.dumps({"success": True, "data": {"value": "1"}}))
        assert res_rec is None
        
        res_list = brain._build_grounded_response("list_memory", {}, json.dumps({"success": True, "data": {"count": 1}}))
        assert res_list is None

    @patch("core.memory.remember")
    def test_remember_pipeline(self, mock_rem):
        mock_rem.return_value = {"success": True, "data": {"key": "x", "value": "y"}}
        
        brain = _make_brain('{"tool": "remember", "args": {"key": "x", "value": "y"}}')
        response = brain.process_input("Remember that x is y")
        
        mock_rem.assert_called_once_with(key="x", value="y")
        # LLM Pass 2 bypassed because response is deterministic
        brain.llm.chat.assert_not_called()
        assert "Done!" in response


class TestRegressionGuardPhase19:
    def test_existing_tools_still_registered(self):
        registry = ToolRegistry()
        # 33(p16) + 3(p17) + 1(p18) + 4(p19) = 41
        assert len(registry.tools) >= 41

    def test_write_file_still_dangerous(self):
        registry = ToolRegistry()
        assert registry.tools["write_file"][1] is True
