import os
import tempfile
import pytest
from skills import file_ops
from core.brain import Brain
from core.tool_registry import ToolRegistry

@pytest.fixture
def temp_env(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv('USERPROFILE', temp_dir)
        # Create Desktop, Documents, Downloads so they exist when needed
        os.makedirs(os.path.join(temp_dir, 'Desktop'))
        os.makedirs(os.path.join(temp_dir, 'Documents'))
        os.makedirs(os.path.join(temp_dir, 'Downloads'))
        yield temp_dir

def test_resolve_secure_path_success(temp_env):
    path = file_ops._resolve_secure_path("desktop", "test_folder")
    assert path == os.path.join(temp_env, "Desktop", "test_folder")
    
    path = file_ops._resolve_secure_path("documents")
    assert path == os.path.join(temp_env, "Documents")

def test_resolve_secure_path_rejection(temp_env):
    with pytest.raises(ValueError, match="Location not permitted"):
        file_ops._resolve_secure_path("windows", "test")
        
    with pytest.raises(ValueError, match="Name cannot contain path separators"):
        file_ops._resolve_secure_path("desktop", "folder/subfolder")
        
    with pytest.raises(ValueError, match="Name cannot contain path separators"):
        file_ops._resolve_secure_path("desktop", "..\\folder")

    with pytest.raises(ValueError, match="Invalid name"):
        file_ops._resolve_secure_path("desktop", "..")

def test_create_folder_success(temp_env):
    res = file_ops.create_folder("desktop", "Projects")
    assert res["success"] is True
    assert os.path.isdir(os.path.join(temp_env, "Desktop", "Projects"))

def test_create_folder_existing(temp_env):
    file_ops.create_folder("desktop", "Projects")
    res = file_ops.create_folder("desktop", "Projects")
    assert res["success"] is False
    assert "already exists" in res["error"]

def test_create_file_success(temp_env):
    res = file_ops.create_file("documents", "notes.txt")
    assert res["success"] is True
    assert os.path.isfile(os.path.join(temp_env, "Documents", "notes.txt"))

def test_create_file_existing(temp_env):
    file_ops.create_file("documents", "notes.txt")
    res = file_ops.create_file("documents", "notes.txt")
    assert res["success"] is False
    assert "already exists" in res["error"]

def test_list_directory_success(temp_env):
    file_ops.create_file("downloads", "file1.txt")
    file_ops.create_folder("downloads", "folder1")
    res = file_ops.list_directory("downloads")
    assert res["success"] is True
    assert "file1.txt" in res["data"]["items"]
    assert "folder1" in res["data"]["items"]

def test_list_directory_empty(temp_env):
    res = file_ops.list_directory("downloads")
    assert res["success"] is True
    assert len(res["data"]["items"]) == 0

def test_open_file_or_folder_success(temp_env, monkeypatch):
    # Mock os.startfile so it doesn't actually open anything during test
    calls = []
    monkeypatch.setattr(os, "startfile", lambda p: calls.append(p))
    
    file_ops.create_file("desktop", "notes.txt")
    res = file_ops.open_file_or_folder("desktop", "notes.txt")
    assert res["success"] is True
    assert len(calls) == 1
    assert "notes.txt" in calls[0]

def test_tool_registry_no_delete():
    registry = ToolRegistry()
    assert "delete_path" not in registry.tools
    assert "create_folder" in registry.tools
    assert "create_file" in registry.tools
    assert "list_directory" in registry.tools
    assert "open_file_or_folder" in registry.tools

def test_brain_grounding_list_directory(temp_env):
    brain = Brain()
    # test list_directory grounding
    file_ops.create_file("desktop", "file1.txt")
    file_ops.create_folder("desktop", "dir1")
    
    # simulate tool registry execution
    result = brain.registry.execute_tool("list_directory", {"location": "desktop"})
    grounded = brain._build_grounded_response("list_directory", {"location": "desktop"}, result)
    assert "file1.txt" in grounded
    assert "dir1" in grounded
