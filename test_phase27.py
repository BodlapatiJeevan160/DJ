"""
test_phase27.py
===============
Phase 27: Persistent Conversation & Session History

Tests cover:
  - SessionStore unit tests (fresh startup, restore, trimming, size cap,
    atomic persistence, malformed JSON recovery, clear, concurrency)
  - Brain integration tests (init restore, per-turn save, fast-path routing,
    deterministic response, restart simulation)
  - Phase 27 isolation from Phase 19 (key-value memory) and Phase 25
    (semantic vector memory)
  - Error resilience (read-only directory, oversized file)
  - Credential sanitization
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turns(n: int, start_with_system: bool = True) -> list:
    """
    Constructs a synthetic history list of *n* messages.
    If start_with_system, prepends a system message so the total
    list length = n + 1.
    """
    history = []
    if start_with_system:
        history.append({"role": "system", "content": "You are JARVIS."})
    for i in range(n):
        history.append({"role": "user",      "content": f"User message {i}"})
        history.append({"role": "assistant", "content": f"Assistant reply {i}"})
    return history


# ---------------------------------------------------------------------------
# Base class that redirects SessionStore to a temp directory for every test
# ---------------------------------------------------------------------------

class _SessionStoreTestBase(unittest.TestCase):
    """Patches STORAGE_DIR / STORAGE_FILE to a temp directory before each test."""

    def setUp(self):
        import core.session as _sm
        self._sess_module = _sm
        self._test_dir = tempfile.mkdtemp()
        self._orig_dir  = _sm.STORAGE_DIR
        self._orig_file = _sm.STORAGE_FILE
        _sm.STORAGE_DIR  = self._test_dir
        _sm.STORAGE_FILE = os.path.join(self._test_dir, "history.json")
        self._hist_file  = _sm.STORAGE_FILE

    def tearDown(self):
        self._sess_module.STORAGE_DIR  = self._orig_dir
        self._sess_module.STORAGE_FILE = self._orig_file
        shutil.rmtree(self._test_dir, ignore_errors=True)


# ===========================================================================
# T1 – T12: SessionStore unit tests
# ===========================================================================

class TestSessionStoreUnit(_SessionStoreTestBase):

    # -------------------------------------------------------------------------
    # T1: save() with empty list → file created, contains []
    # -------------------------------------------------------------------------
    def test_t01_save_empty_list_creates_file(self):
        from core.session import SessionStore
        result = SessionStore.save([])
        self.assertTrue(result, "save() should succeed for empty list")
        self.assertTrue(os.path.exists(self._hist_file))
        with open(self._hist_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, [])

    # -------------------------------------------------------------------------
    # T2: save() + load() round-trip preserves role / content faithfully
    # -------------------------------------------------------------------------
    def test_t02_save_load_roundtrip(self):
        from core.session import SessionStore
        sample = [
            {"role": "system",    "content": "You are JARVIS."},
            {"role": "user",      "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        SessionStore.save(sample)
        loaded = SessionStore.load()
        self.assertEqual(loaded, sample)

    # -------------------------------------------------------------------------
    # T3: load() with missing file → returns []
    # -------------------------------------------------------------------------
    def test_t03_load_missing_file_returns_empty(self):
        from core.session import SessionStore
        self.assertFalse(os.path.exists(self._hist_file))
        result = SessionStore.load()
        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # T4: load() with corrupt JSON → returns []
    # -------------------------------------------------------------------------
    def test_t04_load_corrupt_json_returns_empty(self):
        from core.session import SessionStore
        with open(self._hist_file, "w", encoding="utf-8") as f:
            f.write("{this is not valid JSON at all!}")
        result = SessionStore.load()
        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # T5: load() with file > 512 KB → returns []
    # -------------------------------------------------------------------------
    def test_t05_load_oversized_file_returns_empty(self):
        from core.session import SessionStore, MAX_FILE_SIZE_BYTES
        with open(self._hist_file, "wb") as f:
            f.write(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        result = SessionStore.load()
        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # T6: load() with non-list JSON → returns []
    # -------------------------------------------------------------------------
    def test_t06_load_non_list_json_returns_empty(self):
        from core.session import SessionStore
        with open(self._hist_file, "w", encoding="utf-8") as f:
            json.dump({"role": "user", "content": "hi"}, f)
        result = SessionStore.load()
        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # T7: load() with list containing a non-dict entry → that entry is skipped
    # -------------------------------------------------------------------------
    def test_t07_load_non_dict_entry_skipped(self):
        from core.session import SessionStore
        raw = [
            {"role": "user", "content": "valid"},
            "this is not a dict",
            42,
            {"role": "assistant", "content": "also valid"},
        ]
        with open(self._hist_file, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        loaded = SessionStore.load()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["content"], "valid")
        self.assertEqual(loaded[1]["content"], "also valid")

    # -------------------------------------------------------------------------
    # T8: save() trims to MAX_TURNS when given a 100-entry history
    # -------------------------------------------------------------------------
    def test_t08_save_trims_to_max_turns(self):
        from core.session import SessionStore, MAX_TURNS
        # 1 system + 49 user/assistant pairs = 1 + 98 = 99 messages > MAX_TURNS (50)
        history = _make_turns(49, start_with_system=True)  # 1 + 98 = 99
        self.assertGreater(len(history), MAX_TURNS)

        SessionStore.save(history)
        loaded = SessionStore.load()

        # Must not exceed MAX_TURNS
        self.assertLessEqual(len(loaded), MAX_TURNS)
        # Initial system message is always preserved
        self.assertEqual(loaded[0]["role"], "system")
        # Most-recent content is preserved (last assistant reply)
        contents = [m["content"] for m in loaded]
        self.assertIn("Assistant reply 48", contents)

    # -------------------------------------------------------------------------
    # T9: save() is atomic — temp file is renamed, never half-written
    # -------------------------------------------------------------------------
    def test_t09_save_is_atomic(self):
        from core.session import SessionStore
        first  = [{"role": "user", "content": "first"}]
        second = [{"role": "user", "content": "second"}]

        SessionStore.save(first)
        # Simulate a concurrent replacement
        SessionStore.save(second)

        loaded = SessionStore.load()
        # File must contain exactly one of the saves, never a mix
        contents = [m["content"] for m in loaded]
        self.assertIn("second", contents)
        self.assertNotIn("first", contents)

        # Additionally verify no temp files were left behind
        leftover = [
            f for f in os.listdir(self._test_dir)
            if f.startswith("hist_") and f.endswith(".tmp")
        ]
        self.assertEqual(leftover, [], "Temporary write files should be cleaned up")

    # -------------------------------------------------------------------------
    # T10: clear() deletes file → subsequent load() returns []
    # -------------------------------------------------------------------------
    def test_t10_clear_deletes_file(self):
        from core.session import SessionStore
        SessionStore.save([{"role": "user", "content": "test"}])
        self.assertTrue(os.path.exists(self._hist_file))

        ok = SessionStore.clear()
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(self._hist_file))
        self.assertEqual(SessionStore.load(), [])

    # -------------------------------------------------------------------------
    # T11: clear() on non-existent file → returns True (idempotent)
    # -------------------------------------------------------------------------
    def test_t11_clear_idempotent(self):
        from core.session import SessionStore
        self.assertFalse(os.path.exists(self._hist_file))
        ok = SessionStore.clear()
        self.assertTrue(ok, "clear() should return True even if file is absent")

    # -------------------------------------------------------------------------
    # T12: Concurrent save() calls from multiple threads → no corruption
    # -------------------------------------------------------------------------
    def test_t12_concurrent_save_no_corruption(self):
        from core.session import SessionStore
        histories = [
            [{"role": "user", "content": f"Thread {i}"}] for i in range(12)
        ]
        threads = [
            threading.Thread(target=SessionStore.save, args=(h,))
            for h in histories
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The file must be valid JSON after all concurrent writes
        result = SessionStore.load()
        self.assertIsInstance(result, list, "Concurrent saves must produce valid JSON")


# ===========================================================================
# T13 – T17: Brain integration tests
# ===========================================================================

class TestBrainIntegration(_SessionStoreTestBase):
    """
    Tests that Brain correctly integrates with SessionStore.
    LLMClient and ToolRegistry are mocked so Ollama is not required.
    """

    def _make_brain(self):
        """Returns a Brain instance with Ollama fully mocked."""
        with patch("core.brain.LLMClient") as mock_llm_cls, \
             patch("core.brain.ToolRegistry") as mock_reg_cls:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = "JARVIS mocked response"
            mock_llm.chat_json.return_value = '{"tool": null, "args": null}'
            mock_llm_cls.return_value = mock_llm

            mock_reg = MagicMock()
            mock_reg.get_tool_descriptions.return_value = ""
            mock_reg.execute_tool.return_value = '{"success": true, "tool": "clear_history", "data": {"cleared": true}, "error": null}'
            mock_reg_cls.return_value = mock_reg

            from core.brain import Brain
            brain = Brain()
            brain._mock_llm = mock_llm
            brain._mock_reg = mock_reg
        return brain

    # -------------------------------------------------------------------------
    # T13: Brain.__init__ calls SessionStore.load() and populates self.history
    # -------------------------------------------------------------------------
    def test_t13_brain_init_restores_history(self):
        from core.session import SessionStore
        sample = [
            {"role": "system",    "content": "You are JARVIS."},
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        SessionStore.save(sample)

        brain = self._make_brain()
        self.assertGreater(len(brain.history), 0, "Brain should restore non-empty history")
        roles = [m["role"] for m in brain.history]
        self.assertIn("user",      roles)
        self.assertIn("assistant", roles)

    # -------------------------------------------------------------------------
    # T14: process_input() persists history (SessionStore.save) after each turn
    # -------------------------------------------------------------------------
    def test_t14_process_input_saves_history(self):
        from core.session import SessionStore
        brain = self._make_brain()

        # Mock the LLM to route conversationally (no tool)
        brain._mock_llm.chat_json.return_value = '{"tool": null, "args": null}'
        brain._mock_llm.chat.return_value = "Hello! How can I help you?"

        with patch.object(SessionStore, "save", wraps=SessionStore.save) as mock_save:
            brain.process_input("Hello JARVIS")
            mock_save.assert_called_once()

    # -------------------------------------------------------------------------
    # T15: Fast-path "clear history" → tool_name dispatched as clear_history
    # -------------------------------------------------------------------------
    def test_t15_fast_path_clear_history(self):
        from core.brain import _CLEAR_HISTORY_PATTERN
        matching = [
            "clear history",
            "Clear History",
            "clear our history",
            "clear conversation history",
            "clear the session history",
            "delete history",
            "erase history",
            "reset history",
        ]
        for phrase in matching:
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(
                    _CLEAR_HISTORY_PATTERN.match(phrase.strip()),
                    f"Expected pattern to match: '{phrase}'"
                )

    # -------------------------------------------------------------------------
    # T16: Fast-path "forget our conversation" → matched by clear_history pattern
    # -------------------------------------------------------------------------
    def test_t16_fast_path_forget_conversation(self):
        from core.brain import _CLEAR_HISTORY_PATTERN
        matching = [
            "forget our conversation",
            "forget conversation",
            "forget our session",
            "forget our session history",
            "clear the conversation",
            "clear the chat",
            "clear session",
        ]
        for phrase in matching:
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(
                    _CLEAR_HISTORY_PATTERN.match(phrase.strip()),
                    f"Expected pattern to match: '{phrase}'"
                )

    # -------------------------------------------------------------------------
    # T15b: Phrases that must NOT match the clear_history pattern
    # -------------------------------------------------------------------------
    def test_t15b_fast_path_no_false_positives(self):
        from core.brain import _CLEAR_HISTORY_PATTERN
        non_matching = [
            "clear browser history",    # browser history — not conversation
            "forget my name",           # specific fact — goes to LLM/memory
            "forget everything",        # too vague — goes to LLM
            "what is my history",       # question, not command
            "show me my history",       # different intent
        ]
        for phrase in non_matching:
            with self.subTest(phrase=phrase):
                self.assertIsNone(
                    _CLEAR_HISTORY_PATTERN.match(phrase.strip()),
                    f"Expected pattern NOT to match: '{phrase}'"
                )

    # -------------------------------------------------------------------------
    # T17: clear_history _build_grounded_response returns deterministic string
    # -------------------------------------------------------------------------
    def test_t17_clear_history_deterministic_response(self):
        brain = self._make_brain()
        result_json = json.dumps({
            "success": True, "tool": "clear_history",
            "data": {"cleared": True}, "error": None
        })
        response = brain._build_grounded_response("clear_history", {}, result_json)
        self.assertIsNotNone(response)
        self.assertIn("cleared", response.lower())
        # Must not be a dict or JSON — must be plain text
        self.assertIsInstance(response, str)
        self.assertNotIn("{", response)


# ===========================================================================
# T18 – T22: Phase 27 end-to-end and compatibility tests
# ===========================================================================

class TestPhase27Compatibility(_SessionStoreTestBase):

    def _make_brain(self):
        with patch("core.brain.LLMClient") as mock_llm_cls, \
             patch("core.brain.ToolRegistry") as mock_reg_cls:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = "JARVIS mocked response"
            mock_llm.chat_json.return_value = '{"tool": null, "args": null}'
            mock_llm_cls.return_value = mock_llm

            mock_reg = MagicMock()
            mock_reg.get_tool_descriptions.return_value = ""
            mock_reg_cls.return_value = mock_reg

            from core.brain import Brain
            brain = Brain()
            brain._mock_llm = mock_llm
            brain._mock_reg = mock_reg
        return brain

    # -------------------------------------------------------------------------
    # T18: History is restored on Brain re-instantiation (simulate restart)
    # -------------------------------------------------------------------------
    def test_t18_history_restored_on_reinit(self):
        from core.session import SessionStore

        # Manually save a history simulating the end of a prior session
        prior_history = [
            {"role": "system",    "content": "You are JARVIS."},
            {"role": "user",      "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
        ]
        SessionStore.save(prior_history)

        # Simulate a restart by creating a fresh Brain (loads history.json)
        brain = self._make_brain()
        contents = [m["content"] for m in brain.history]
        self.assertIn("My name is Alice.",       contents)
        self.assertIn("Nice to meet you, Alice!", contents)

    # -------------------------------------------------------------------------
    # T19: Session history does NOT contain Phase 19 key-value memory contents
    # -------------------------------------------------------------------------
    def test_t19_isolation_from_phase19_kv_memory(self):
        from core.session import SessionStore

        history = [
            {"role": "system",    "content": "You are JARVIS."},
            {"role": "user",      "content": "Remember my favorite color is blue."},
            {"role": "assistant", "content": "Noted."},
        ]
        SessionStore.save(history)

        # Phase 19 memory.json path
        phase19_memory_path = os.path.join(
            os.path.expanduser("~"), ".jarvis", "memory.json"
        )
        # Verify the saved history file is different from memory.json
        self.assertNotEqual(
            os.path.realpath(self._hist_file),
            os.path.realpath(phase19_memory_path),
        )

        # Verify loaded history is purely the conversation; no memory.json injection
        loaded = SessionStore.load()
        all_content = " ".join(m["content"] for m in loaded)
        # The session content should be exactly what we saved
        self.assertIn("favorite color is blue", all_content)
        # It should NOT contain Phase 19 memory keys/values in injected form
        # (those come from _get_memory_block at Pass-2 time, not here)

    # -------------------------------------------------------------------------
    # T20: Session history does NOT reference ChromaDB (Phase 25) storage paths
    # -------------------------------------------------------------------------
    def test_t20_isolation_from_phase25_semantic_memory(self):
        from core.session import SessionStore

        history = [
            {"role": "system",    "content": "You are JARVIS."},
            {"role": "user",      "content": "Store: I love hiking."},
            {"role": "assistant", "content": "Saved to semantic memory."},
        ]
        SessionStore.save(history)
        loaded = SessionStore.load()

        # The history.json storage path must not be inside the chroma_db directory
        chroma_dir = os.path.join(os.path.expanduser("~"), ".jarvis", "chroma_db")
        self.assertFalse(
            self._hist_file.startswith(chroma_dir),
            "history.json must be outside the ChromaDB directory",
        )
        # Loaded content must match what was saved — no ChromaDB injection
        self.assertEqual(len(loaded), 3)

    # -------------------------------------------------------------------------
    # T21: save() on a read-only directory → returns False without crashing JARVIS
    # -------------------------------------------------------------------------
    def test_t21_save_readonly_directory_no_crash(self):
        from core.session import SessionStore
        from unittest.mock import patch
        
        # Simulate a PermissionError when trying to write to the storage directory
        with patch("core.session.os.makedirs", side_effect=PermissionError("Access denied")):
            history = [{"role": "user", "content": "test"}]
            result  = SessionStore.save(history)
            # Must return False (failure) without raising
            self.assertFalse(result, "save() must return False on I/O failure")

    # -------------------------------------------------------------------------
    # T22: Restored history with ACTION EXECUTED entries is structurally valid
    #      for Pass-2 grounding (entries survive the load/validate cycle)
    # -------------------------------------------------------------------------
    def test_t22_restored_history_preserves_action_executed_records(self):
        from core.session import SessionStore

        history = [
            {"role": "system",    "content": "You are JARVIS."},
            {"role": "user",      "content": "Search the web for Python."},
            {"role": "system",    "content": "ACTION EXECUTED: search_web\nRESULT: {\"success\": true}"},
            {"role": "assistant", "content": "Here are the results..."},
        ]
        SessionStore.save(history)
        loaded = SessionStore.load()

        # All four entries must survive validation
        self.assertEqual(len(loaded), 4)
        # The ACTION EXECUTED system message is present
        action_msg = next(
            (m for m in loaded
             if m["role"] == "system" and "ACTION EXECUTED" in m["content"]),
            None,
        )
        self.assertIsNotNone(
            action_msg,
            "ACTION EXECUTED records must be preserved by load() for Pass-2 grounding",
        )


# ===========================================================================
# T23: Credential sanitization
# ===========================================================================

class TestCredentialSanitization(_SessionStoreTestBase):

    def test_t23_credential_values_are_redacted_on_save(self):
        from core.session import SessionStore, _sanitize_content

        sensitive_phrases = [
            ("api_key=ABCD1234EFGHijkl",       "api_key=",     "[REDACTED]"),
            ("password: SuperSecret123!",        "password: ",   "[REDACTED]"),
            ("secret_key = abc123xyz789",        "secret_key",   "[REDACTED]"),
            ("Bearer eyJhbGciOiJIUzI1NiJ9abc",  "Bearer ",      "[REDACTED]"),
            ("Authorization: Basic dXNlcjpwYXNz", "Basic ",     "[REDACTED]"),
        ]
        for raw, preserved_prefix, expected_replacement in sensitive_phrases:
            with self.subTest(raw=raw):
                result = _sanitize_content(raw)
                self.assertIn(
                    "[REDACTED]", result,
                    f"Expected [REDACTED] in sanitized output for: {raw!r}"
                )
                # The key name / prefix must be preserved
                self.assertIn(
                    preserved_prefix.lower(),
                    result.lower(),
                    f"Key prefix should be preserved in: {result!r}"
                )

    def test_t23b_normal_content_is_unmodified(self):
        from core.session import _sanitize_content

        normal_phrases = [
            "What time is it?",
            "Open Notepad",
            "Set volume to 50",
            "My favorite color is blue",
            "Search the web for Python tutorials",
            "",
        ]
        for phrase in normal_phrases:
            with self.subTest(phrase=phrase):
                result = _sanitize_content(phrase)
                self.assertEqual(
                    result, phrase,
                    f"Normal content must be returned unchanged: {phrase!r}"
                )

    def test_t23c_short_values_not_redacted(self):
        """Values shorter than 8 chars should not be redacted (too many false-positives)."""
        from core.session import _sanitize_content

        short_value = "password=abc"  # value "abc" is 3 chars < 8
        result = _sanitize_content(short_value)
        # Should not contain [REDACTED] since value is too short to be a secret
        self.assertNotIn("[REDACTED]", result)


# ===========================================================================
# T24: SessionStore.is_first_use() behaviour
# ===========================================================================

class TestFirstUseBehaviour(_SessionStoreTestBase):

    def test_t24_is_first_use_true_when_no_file(self):
        from core.session import SessionStore
        self.assertFalse(os.path.exists(self._hist_file))
        self.assertTrue(SessionStore.is_first_use())

    def test_t24b_is_first_use_false_after_save(self):
        from core.session import SessionStore
        SessionStore.save([])
        self.assertFalse(SessionStore.is_first_use())

    def test_t24c_is_first_use_true_after_clear(self):
        from core.session import SessionStore
        SessionStore.save([{"role": "user", "content": "hi"}])
        SessionStore.clear()
        self.assertTrue(SessionStore.is_first_use())


# ===========================================================================
# T25: History trimming edge cases
# ===========================================================================

class TestHistoryTrimming(_SessionStoreTestBase):

    def test_t25_history_with_no_leading_system_trimmed_correctly(self):
        from core.session import SessionStore, MAX_TURNS

        # Build history without a leading system message
        history = [
            {"role": "user",      "content": f"msg {i}"}
            for i in range(MAX_TURNS + 10)
        ]
        SessionStore.save(history)
        loaded = SessionStore.load()
        self.assertLessEqual(len(loaded), MAX_TURNS)
        # Most recent messages are kept
        self.assertEqual(loaded[-1]["content"], f"msg {MAX_TURNS + 10 - 1}")

    def test_t25b_history_at_exact_max_turns_not_trimmed(self):
        from core.session import SessionStore, MAX_TURNS

        history = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg {i}"}
            for i in range(MAX_TURNS - 1)
        ]
        self.assertEqual(len(history), MAX_TURNS)
        SessionStore.save(history)
        loaded = SessionStore.load()
        self.assertEqual(len(loaded), MAX_TURNS)

    def test_t25c_invalid_role_rejected_on_load(self):
        from core.session import SessionStore

        raw = [
            {"role": "user",    "content": "valid"},
            {"role": "unknown", "content": "should be rejected"},
            {"role": "tool",    "content": "also rejected"},
        ]
        with open(self._hist_file, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        loaded = SessionStore.load()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["content"], "valid")

    def test_t25d_oversized_content_entry_rejected(self):
        from core.session import SessionStore, _MAX_CONTENT_LEN

        raw = [
            {"role": "user",      "content": "ok"},
            {"role": "assistant", "content": "x" * (_MAX_CONTENT_LEN + 1)},
        ]
        with open(self._hist_file, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        loaded = SessionStore.load()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["content"], "ok")

    def test_t25e_extra_keys_stripped_on_load(self):
        """Entries with extra unexpected keys are accepted but only role/content kept."""
        from core.session import SessionStore

        raw = [
            {"role": "user", "content": "hi", "extra_field": "malicious", "timestamp": 12345},
        ]
        with open(self._hist_file, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        loaded = SessionStore.load()
        self.assertEqual(len(loaded), 1)
        self.assertNotIn("extra_field", loaded[0])
        self.assertNotIn("timestamp",   loaded[0])
        self.assertEqual(loaded[0]["role"],    "user")
        self.assertEqual(loaded[0]["content"], "hi")


# ===========================================================================
# T26: clear_history tool function
# ===========================================================================

class TestClearHistoryTool(_SessionStoreTestBase):

    def test_t26_clear_history_tool_success(self):
        from core.session import SessionStore, clear_history
        SessionStore.save([{"role": "user", "content": "test"}])
        result = clear_history()
        self.assertTrue(result["success"])
        self.assertEqual(result["tool"], "clear_history")
        self.assertFalse(os.path.exists(self._hist_file))

    def test_t26b_clear_history_tool_idempotent(self):
        from core.session import clear_history
        # Calling when no file exists should still succeed
        result = clear_history()
        self.assertTrue(result["success"])

    def test_t26c_clear_history_registered_in_tool_registry(self):
        """clear_history must be registered and non-dangerous."""
        from core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        self.assertIn("clear_history", registry.tools)
        _, is_dangerous = registry.tools["clear_history"]
        self.assertFalse(is_dangerous, "clear_history must NOT require confirmation")


# ===========================================================================
# T27: Pre-Classifier Routing Tests (Phase 27 Hardening)
# ===========================================================================

class TestRoutingPreClassifier(unittest.TestCase):

    def setUp(self):
        from core.brain import Brain
        self.brain = Brain()
        self.brain.llm = MagicMock()
        self.brain.llm.chat_json.return_value = '{"tool": null, "args": null}'
        self.brain.llm.chat.return_value = "Mock response"
        self.brain.registry.execute_tool = MagicMock(return_value='{"success": true, "data": {}}')
        
        # Patch save to avoid serialization errors with MagicMock
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def _assert_skip(self, user_input: str):
        """Asserts that the LLM is NOT called for obvious conversation."""
        self.brain.llm.chat_json.reset_mock()
        self.brain.registry.execute_tool.reset_mock()
        
        self.brain.process_input(user_input)
        
        self.brain.llm.chat_json.assert_not_called()
        self.brain.registry.execute_tool.assert_not_called()

    def _assert_tool_routed(self, user_input: str, mock_tool_response=None):
        """Asserts that a tool is routed to (either via fast-path or pass1 LLM)."""
        self.brain.llm.chat_json.reset_mock()
        self.brain.registry.execute_tool.reset_mock()
        
        if mock_tool_response:
            self.brain.llm.chat_json.return_value = mock_tool_response
            
        self.brain.process_input(user_input)
        
        # It should hit execute_tool (either deterministic or LLM chosen)
        self.brain.registry.execute_tool.assert_called()

    def test_t27a_conversation_bypass(self):
        # ordinary statement
        self._assert_skip("The weather is nice today.")
        self._assert_skip("My favorite color is blue.")
        # ordinary question
        self._assert_skip("How are you doing today?")
        # previous-conversation question
        self._assert_skip("What was the unique session marker I gave you?")
        self._assert_skip("Do you remember what we talked about?")
        # summary request
        self._assert_skip("Summarize what we discussed.")
        self._assert_skip("Can you summarize our conversation?")
        # general knowledge question
        self._assert_skip("Explain what Python is.")
        self._assert_skip("What is the capital of France?")

    def test_t27b_tool_routing_still_works(self):
        # list_windows
        self._assert_tool_routed("List my open windows", '{"tool": "list_windows", "args": {}}')
        self._assert_tool_routed("Show all open windows") # hits deterministic
        # search_workspace
        self._assert_tool_routed("Search my workspace for MAX_MEMORIES", '{"tool": "search_workspace", "args": {"query": "MAX_MEMORIES"}}')
        self._assert_tool_routed("Search the codebase for session", '{"tool": "search_workspace", "args": {"query": "session"}}')
        # remember
        self._assert_tool_routed("Remember that my favorite language is Python", '{"tool": "remember", "args": {"key": "lang", "value": "Python"}}')
        # file operation
        self._assert_tool_routed("Create a file called test.txt", '{"tool": "create_file", "args": {"path": "test.txt", "content": ""}}')
        self._assert_tool_routed("List files in Downloads", '{"tool": "list_directory", "args": {"path": "Downloads"}}')
        # reminder
        self._assert_tool_routed("Set a reminder for 10 minutes", '{"tool": "reminder", "args": {"time": "10m"}}')
        self._assert_tool_routed("Remind me to check the oven", '{"tool": "reminder", "args": {"message": "check the oven"}}')
        # browser/search tool
        self._assert_tool_routed("Open Chrome", '{"tool": "open_application", "args": {"application": "chrome"}}')
        self._assert_tool_routed("Search the web for news", '{"tool": "search_web", "args": {"query": "news"}}')

    def test_t27c_safety_guards_not_bypassed(self):
        # analyze_screen guard
        self.brain.llm.chat_json.return_value = '{"tool": "analyze_screen", "args": {}}'
        self.brain.registry.execute_tool.reset_mock()
        self.brain.process_input("Open this file please")
        
        # Because "Open this file please" lacks screen intent, the guard should prevent execute_tool
        self.brain.registry.execute_tool.assert_not_called()

# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
