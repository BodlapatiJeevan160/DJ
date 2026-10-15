import os
import tempfile
import json
import pytest
from skills import file_ops
from core.brain import Brain
from core.tool_registry import ToolRegistry

@pytest.fixture
def temp_env(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv('USERPROFILE', temp_dir)
        os.makedirs(os.path.join(temp_dir, 'Desktop'))
        os.makedirs(os.path.join(temp_dir, 'Documents'))
        os.makedirs(os.path.join(temp_dir, 'Downloads'))
        yield temp_dir

def test_read_file_success(temp_env):
    path = os.path.join(temp_env, 'Desktop', 'test.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Hello JARVIS")
        
    res = file_ops.read_file("desktop", "test.txt")
    assert res["success"] is True
    assert res["data"]["content"] == "Hello JARVIS"

def test_read_file_not_found(temp_env):
    res = file_ops.read_file("desktop", "missing.txt")
    assert res["success"] is False
    assert "does not exist" in res["error"]

def test_read_file_too_large(temp_env):
    path = os.path.join(temp_env, 'Documents', 'large.txt')
    with open(path, 'wb') as f:
        f.seek((50 * 1024) + 10)
        f.write(b'\0')
        
    res = file_ops.read_file("documents", "large.txt")
    assert res["success"] is False
    assert "too large" in res["error"]

def test_read_file_binary(temp_env):
    path = os.path.join(temp_env, 'Downloads', 'image.png')
    with open(path, 'wb') as f:
        # Invalid UTF-8 sequence
        f.write(b'\xff\xfe\x00\x00\x89PNG\r\n\x1a\n')
        
    res = file_ops.read_file("downloads", "image.png")
    assert res["success"] is False
    assert "binary or not UTF-8" in res["error"]

def test_read_file_path_traversal(temp_env):
    res = file_ops.read_file("desktop", "../test.txt")
    assert res["success"] is False
    assert "cannot contain path separators" in res["error"]

def test_brain_grounding_read_file(temp_env):
    brain = Brain()
    result = json.dumps({"success": True, "tool": "read_file", "data": {"content": "Secret info"}, "error": None})
    grounded = brain._build_grounded_response("read_file", {"location": "desktop", "name": "secret.txt"}, result)
    # Phase 29: read_file success now returns a deterministic string (no LLM Pass 2)
    assert grounded is not None
    assert "Contents of secret.txt:" in grounded
    assert "Secret info" in grounded
    
    # On failure, it should return deterministic string
    result_fail = json.dumps({"success": False, "tool": "read_file", "data": None, "error": "File does not exist."})
    grounded_fail = brain._build_grounded_response("read_file", {"location": "desktop", "name": "missing.txt"}, result_fail)
    assert "failed: File does not exist" in grounded_fail

def test_tool_registry_read_file():
    registry = ToolRegistry()
    assert "read_file" in registry.tools
