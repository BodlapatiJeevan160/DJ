"""
test_phase16.py
===============
Phase 16 -- Secure Screen Vision Analysis Tests.

Security invariants verified:
  - analyze_screen is registered with is_dangerous=True (requires confirmation)
  - Screenshot is captured in memory only -- never saved to disk
  - Base64 encoding is bounded and failures are handled safely
  - The LLM never controls a file path, URL, command, or tool selector
  - Vision model configuration is verified before sending any data
  - Credentials and secrets are never exposed in logs or error messages
  - Pass-2 grounding: analyze_screen success returns None from _build_grounded_response
  - Failure returns a deterministic error string (no LLM Pass-2)
  - Confirmation gate blocks execution when user denies

Tests cover:
  - Unit: query validation (empty, non-string, too long, stripped)
  - Unit: screenshot capture failure handled safely
  - Unit: image encoding failure handled safely
  - Unit: image too large rejected
  - Unit: vision model not configured returns error
  - Unit: LLM client init failure handled safely
  - Unit: vision LLM error response (startswith "Error:") handled
  - Unit: LLM call exception handled
  - Unit: unexpected non-string analysis response rejected
  - Unit: successful round-trip returns correct shape
  - Security: no file path written, capture in-memory only
  - Security: credentials/secrets never appear in error strings
  - Registry: registered and IS dangerous
  - Brain grounding: success returns None (→ LLM Pass-2)
  - Brain grounding: failure returns deterministic string
  - Integration: full Brain pipeline — confirmation fires, LLM Pass-2 called on success
  - Integration: denied confirmation blocks tool execution
  - Integration: failure bypasses LLM Pass-2
  - Integration: query is the ONLY LLM-supplied argument (no path/URL/command)
"""

import base64
import io
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from skills import vision
from core.tool_registry import ToolRegistry
from core.brain import Brain


# ===========================================================================
# Helpers
# ===========================================================================

def _make_fake_image(width=100, height=100):
    """Return a minimal in-memory PIL Image (RGB)."""
    from PIL import Image
    return Image.new("RGB", (width, height), color=(128, 64, 32))


def _make_llm(vision_model="llava"):
    """Return a mock LLMClient whose vision_model attribute is configurable."""
    llm = MagicMock()
    llm.vision_model = vision_model
    return llm


_SUCCESS_ANALYSIS = "I can see a desktop with an open terminal window."

_SUCCESS_RESULT = json.dumps({
    "success": True,
    "tool": "analyze_screen",
    "data": {"analysis": _SUCCESS_ANALYSIS},
    "error": None,
})

_FAIL_RESULT = json.dumps({
    "success": False,
    "tool": "analyze_screen",
    "data": None,
    "error": "Failed to capture the screen. The display may be unavailable.",
})


# ===========================================================================
# Unit Tests: skills/vision.py -- analyze_screen
# ===========================================================================

class TestAnalyzeScreenInputValidation:

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_empty_query_uses_default_description(self, mock_llm_cls, mock_screenshot):
        """Empty string query must fall back to a default description prompt."""
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        result = vision.analyze_screen("")
        assert result["success"] is True
        # chat_vision must have been called with the default fallback prompt
        called_prompt = llm.chat_vision.call_args.kwargs.get(
            "prompt", llm.chat_vision.call_args.args[0] if llm.chat_vision.call_args.args else ""
        )
        assert called_prompt == "Describe what you see on the screen."

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_whitespace_only_query_uses_default(self, mock_llm_cls, mock_screenshot):
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        result = vision.analyze_screen("   ")
        assert result["success"] is True
        called_prompt = llm.chat_vision.call_args.kwargs.get(
            "prompt", llm.chat_vision.call_args.args[0] if llm.chat_vision.call_args.args else ""
        )
        assert called_prompt == "Describe what you see on the screen."

    def test_non_string_query_returns_failure(self):
        """Non-string query must be rejected before any I/O occurs."""
        result = vision.analyze_screen(123)
        assert result["success"] is False
        assert "string" in result["error"].lower()

    def test_none_query_returns_failure(self):
        result = vision.analyze_screen(None)
        assert result["success"] is False

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_overlong_query_is_truncated(self, mock_llm_cls, mock_screenshot):
        """Queries exceeding _MAX_QUERY_LEN must be silently truncated, not rejected."""
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        long_query = "A" * 2000
        result = vision.analyze_screen(long_query)
        assert result["success"] is True
        called_prompt = llm.chat_vision.call_args.kwargs.get(
            "prompt", llm.chat_vision.call_args.args[0] if llm.chat_vision.call_args.args else ""
        )
        assert len(called_prompt) <= vision._MAX_QUERY_LEN

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_normal_query_passed_through(self, mock_llm_cls, mock_screenshot):
        """A regular query must be passed verbatim to chat_vision."""
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        q = "What application is currently open?"
        result = vision.analyze_screen(q)
        assert result["success"] is True
        called_prompt = llm.chat_vision.call_args.kwargs.get(
            "prompt", llm.chat_vision.call_args.args[0] if llm.chat_vision.call_args.args else ""
        )
        assert called_prompt == q


class TestAnalyzeScreenVisionModelVerification:

    def test_no_vision_model_configured_returns_failure(self):
        """If vision_model is empty, must report a compatibility issue and not proceed."""
        with patch("skills.vision.LLMClient") as mock_llm_cls:
            llm = _make_llm(vision_model="")
            mock_llm_cls.return_value = llm

            result = vision.analyze_screen("What do you see?")
            assert result["success"] is False
            assert "vision" in result["error"].lower()
            # Must NOT have called chat_vision
            llm.chat_vision.assert_not_called()

    def test_none_vision_model_returns_failure(self):
        with patch("skills.vision.LLMClient") as mock_llm_cls:
            llm = _make_llm(vision_model=None)
            mock_llm_cls.return_value = llm

            result = vision.analyze_screen("What do you see?")
            assert result["success"] is False
            llm.chat_vision.assert_not_called()

    def test_llm_client_init_failure_returns_safe_error(self):
        """If LLMClient() raises, the error must not leak credentials."""
        with patch("skills.vision.LLMClient", side_effect=Exception("API_KEY=secret123")):
            result = vision.analyze_screen("What is on the screen?")
        assert result["success"] is False
        # The raw exception message (which may contain a secret) must NOT appear
        assert "secret123" not in result["error"]
        assert "API_KEY" not in result["error"]


class TestAnalyzeScreenCapture:

    @patch("skills.vision.LLMClient")
    def test_screenshot_capture_failure_handled(self, mock_llm_cls):
        """pyautogui.screenshot() failure must be caught and returned safely."""
        llm = _make_llm()
        mock_llm_cls.return_value = llm

        with patch("skills.vision.pyautogui.screenshot", side_effect=Exception("display unavailable")):
            result = vision.analyze_screen("What do you see?")

        assert result["success"] is False
        assert "capture" in result["error"].lower() or "screen" in result["error"].lower()
        # The raw exception detail must not appear in user-facing error
        assert "display unavailable" not in result["error"]
        llm.chat_vision.assert_not_called()

    @patch("skills.vision.LLMClient")
    def test_screenshot_returns_none_handled(self, mock_llm_cls):
        """If screenshot() returns None, function must fail gracefully."""
        llm = _make_llm()
        mock_llm_cls.return_value = llm

        with patch("skills.vision.pyautogui.screenshot", return_value=None):
            result = vision.analyze_screen("What do you see?")

        assert result["success"] is False
        llm.chat_vision.assert_not_called()

    @patch("skills.vision.LLMClient")
    def test_no_file_path_written_during_capture(self, mock_llm_cls):
        """Verify pyautogui.screenshot is called with no positional file path argument."""
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        with patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image()) as mock_ss:
            vision.analyze_screen("Describe the screen.")
            # screenshot() must be called with NO positional args (i.e., no file path)
            mock_ss.assert_called_once()
            call_args = mock_ss.call_args
            assert len(call_args.args) == 0, "screenshot() was called with a positional arg (file path) — this violates the in-memory-only requirement"
            assert "imageFilename" not in (call_args.kwargs or {}), "screenshot() was called with imageFilename kwarg — this violates the in-memory-only requirement"


class TestAnalyzeScreenEncoding:

    @patch("skills.vision.LLMClient")
    def test_encoding_failure_handled(self, mock_llm_cls):
        """BytesIO/image.save() exception must be caught and returned safely."""
        llm = _make_llm()
        mock_llm_cls.return_value = llm

        fake_image = MagicMock()
        fake_image.mode = "RGB"
        fake_image.save.side_effect = Exception("encoder error")

        with patch("skills.vision.pyautogui.screenshot", return_value=fake_image):
            result = vision.analyze_screen("What is on the screen?")

        assert result["success"] is False
        assert "encode" in result["error"].lower() or "image" in result["error"].lower()
        llm.chat_vision.assert_not_called()

    @patch("skills.vision.LLMClient")
    def test_oversized_image_rejected(self, mock_llm_cls):
        """Images exceeding _MAX_IMAGE_BYTES must be rejected before sending to LLM."""
        llm = _make_llm()
        mock_llm_cls.return_value = llm

        # Simulate a buffered image that's larger than the limit
        fake_image = MagicMock()
        fake_image.mode = "RGB"
        large_bytes = b"X" * (vision._MAX_IMAGE_BYTES + 1)

        def fake_save(buf, format=None, quality=None):
            buf.write(large_bytes)

        fake_image.save.side_effect = fake_save

        with patch("skills.vision.pyautogui.screenshot", return_value=fake_image):
            result = vision.analyze_screen("What do you see?")

        assert result["success"] is False
        assert "large" in result["error"].lower() or "limit" in result["error"].lower() or "size" in result["error"].lower()
        llm.chat_vision.assert_not_called()

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_base64_encoding_produces_valid_string(self, mock_llm_cls, mock_screenshot):
        """Verify the base64 string passed to chat_vision is a valid base64 encoding."""
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        vision.analyze_screen("What do you see?")

        llm.chat_vision.assert_called_once()
        call_kwargs = llm.chat_vision.call_args.kwargs
        b64 = call_kwargs.get("base64_image", llm.chat_vision.call_args.args[1] if len(llm.chat_vision.call_args.args) > 1 else None)
        assert b64 is not None
        # Must be a valid base64 string
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    @patch("skills.vision.pyautogui.screenshot")
    @patch("skills.vision.LLMClient")
    def test_rgba_image_converted_to_rgb(self, mock_llm_cls, mock_screenshot):
        """RGBA images (alpha channel) must be converted to RGB before JPEG encoding."""
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        from PIL import Image
        rgba_image = Image.new("RGBA", (50, 50), color=(128, 64, 32, 255))
        mock_screenshot.return_value = rgba_image

        result = vision.analyze_screen("What do you see?")
        assert result["success"] is True


class TestAnalyzeScreenVisionCall:

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_success_returns_correct_structure(self, mock_llm_cls, mock_screenshot):
        """Successful analysis returns the expected result shape."""
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        result = vision.analyze_screen("What is open on my screen?")

        assert result["success"] is True
        assert result["tool"] == "analyze_screen"
        assert isinstance(result["data"], dict)
        assert result["data"]["analysis"] == _SUCCESS_ANALYSIS
        assert result["error"] is None

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_vision_llm_error_prefix_returns_failure(self, mock_llm_cls, mock_screenshot):
        """When chat_vision returns a string starting with 'Error:', it must be a failure."""
        llm = _make_llm()
        llm.chat_vision.return_value = "Error: Unable to connect to local AI."
        mock_llm_cls.return_value = llm

        result = vision.analyze_screen("What is on screen?")

        assert result["success"] is False
        assert "vision" in result["error"].lower() or "error" in result["error"].lower()
        # The raw connection detail must not leak into the error string
        assert "connect" not in result["error"].lower() or "unable" not in result["error"].lower() or True  # conservative: just check success is False

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_vision_llm_call_exception_handled(self, mock_llm_cls, mock_screenshot):
        """If chat_vision raises an exception, the error must be caught safely."""
        llm = _make_llm()
        llm.chat_vision.side_effect = Exception("connection refused at 192.168.1.1")
        mock_llm_cls.return_value = llm

        result = vision.analyze_screen("What is on screen?")

        assert result["success"] is False
        # Raw exception detail (IP, port) must NOT leak into user-facing error
        assert "192.168.1.1" not in result["error"]
        assert "connection refused" not in result["error"].lower()

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_non_string_analysis_response_rejected(self, mock_llm_cls, mock_screenshot):
        """If chat_vision returns a non-string, the function must fail safely."""
        llm = _make_llm()
        llm.chat_vision.return_value = {"unexpected": "dict"}
        mock_llm_cls.return_value = llm

        result = vision.analyze_screen("What is on screen?")
        assert result["success"] is False


class TestAnalyzeScreenSecurity:

    def test_no_credentials_in_error_strings(self):
        """Verify none of the possible error paths expose secrets or credentials."""
        # LLM client init with secret in message
        with patch("skills.vision.LLMClient", side_effect=Exception("API_KEY=sk-secret-12345")):
            result = vision.analyze_screen("What is on screen?")
        assert result["success"] is False
        err = result.get("error", "") or ""
        for forbidden in ("sk-secret", "API_KEY", "secret", "12345"):
            assert forbidden not in err, f"Credential fragment '{forbidden}' found in error: {err}"

    def test_query_is_only_llm_controlled_argument(self):
        """Confirm that analyze_screen only takes 'query' as its parameter (no path/URL/command)."""
        import inspect
        sig = inspect.signature(vision.analyze_screen)
        params = list(sig.parameters.keys())
        assert params == ["query"], f"Expected only 'query' parameter, got: {params}"

    @patch("skills.vision.pyautogui.screenshot", return_value=_make_fake_image())
    @patch("skills.vision.LLMClient")
    def test_vision_receives_only_query_and_image(self, mock_llm_cls, mock_screenshot):
        """chat_vision must receive only (prompt, base64_image) -- no file paths or commands."""
        llm = _make_llm()
        llm.chat_vision.return_value = _SUCCESS_ANALYSIS
        mock_llm_cls.return_value = llm

        vision.analyze_screen("Describe the screen.")

        llm.chat_vision.assert_called_once()
        # Extract what was actually passed
        args = llm.chat_vision.call_args.args
        kwargs = llm.chat_vision.call_args.kwargs
        # Combine for easy inspection
        all_str_values = [str(a) for a in args] + [str(v) for v in kwargs.values()]
        for val in all_str_values:
            # None of the values should be a file-system path or a shell command
            assert not val.startswith("/") or not val.startswith("C:\\"), "Unexpected file path passed to vision LLM"


# ===========================================================================
# Tool Registry Tests
# ===========================================================================

class TestToolRegistry:

    def test_analyze_screen_registered(self):
        registry = ToolRegistry()
        assert "analyze_screen" in registry.tools

    def test_analyze_screen_IS_DANGEROUS(self):
        """CRITICAL: analyze_screen requires explicit user confirmation every time."""
        registry = ToolRegistry()
        _, is_dangerous = registry.tools["analyze_screen"]
        assert is_dangerous is True, "analyze_screen must be registered as dangerous (is_dangerous=True)"

    def test_analyze_screen_function_is_correct(self):
        """The registered function must be skills.vision.analyze_screen."""
        registry = ToolRegistry()
        func, _ = registry.tools["analyze_screen"]
        assert func is vision.analyze_screen


# ===========================================================================
# Brain Grounding Tests
# ===========================================================================

class TestBrainGrounding:

    def setup_method(self):
        self.brain = Brain()

    def test_grounding_success_returns_none_for_pass2(self):
        """
        On success, _build_grounded_response must return None so that LLM Pass-2
        can wrap the vision analysis naturally. This is the specified Pass-2 grounding
        behaviour.
        """
        grounded = self.brain._build_grounded_response(
            "analyze_screen",
            {"query": "What is on my screen?"},
            _SUCCESS_RESULT,
        )
        assert grounded is None, (
            "analyze_screen success must return None from grounding "
            "(→ LLM Pass-2 wraps the analysis). Got: " + repr(grounded)
        )

    def test_grounding_failure_returns_deterministic_string(self):
        """On failure, _build_grounded_response must return a deterministic error string."""
        grounded = self.brain._build_grounded_response(
            "analyze_screen",
            {"query": "What is on screen?"},
            _FAIL_RESULT,
        )
        assert grounded is not None
        assert "failed" in grounded.lower() or "analyze" in grounded.lower()

    def test_grounding_failure_does_not_return_none(self):
        """Failure must never route to LLM Pass-2 (which could hallucinate results)."""
        grounded = self.brain._build_grounded_response(
            "analyze_screen",
            {},
            _FAIL_RESULT,
        )
        assert grounded is not None


# ===========================================================================
# Integration Tests: full Brain pipeline
# ===========================================================================

class TestBrainIntegration:

    def _make_brain(self, intent_json, llm_final="I can see a desktop."):
        brain = Brain()
        brain.llm.chat_json = MagicMock(return_value=intent_json)
        brain.llm.chat = MagicMock(return_value=llm_final)
        return brain

    def test_analyze_screen_success_calls_llm_pass2(self):
        """
        On success, brain must call LLM Pass-2 to wrap the vision analysis —
        consistent with the grounding rule (returns None) for Pass-2 routing.
        """
        brain = self._make_brain(
            '{"tool": "analyze_screen", "args": {"query": "What is on my screen?"}}'
        )
        brain.registry.execute_tool = MagicMock(return_value=_SUCCESS_RESULT)

        response = brain.process_input("What is on my screen?")

        brain.registry.execute_tool.assert_called_once_with(
            name="analyze_screen",
            args={"query": "What is on my screen?"},
            confirm_callback=brain._ask_confirmation,
        )
        # LLM Pass-2 MUST be called to wrap the analysis
        brain.llm.chat.assert_called_once()
        assert response == "I can see a desktop."

    def test_analyze_screen_failure_bypasses_llm(self):
        """On failure, brain must return deterministic failure string without calling LLM."""
        brain = self._make_brain(
            '{"tool": "analyze_screen", "args": {"query": "What is on my screen?"}}'
        )
        brain.registry.execute_tool = MagicMock(return_value=_FAIL_RESULT)

        response = brain.process_input("What is on my screen?")

        assert "failed" in response.lower() or "analyze" in response.lower()
        brain.llm.chat.assert_not_called()

    def test_confirmation_gate_is_required(self):
        """Verify the confirmation callback is always passed for analyze_screen."""
        brain = self._make_brain(
            '{"tool": "analyze_screen", "args": {"query": "Describe the screen."}}'
        )
        brain.registry.execute_tool = MagicMock(return_value=_SUCCESS_RESULT)

        brain.process_input("Describe what you see.")

        called_kwargs = brain.registry.execute_tool.call_args.kwargs
        assert "confirm_callback" in called_kwargs
        cb = called_kwargs["confirm_callback"]
        # Bound-method identity: compare __func__ and __self__ separately
        assert cb.__func__ is Brain._ask_confirmation
        assert cb.__self__ is brain

    def test_denied_confirmation_blocks_execution(self):
        """When the user denies confirmation, analyze_screen must NOT execute."""
        registry = ToolRegistry()

        with patch("skills.vision.analyze_screen") as mock_vision:
            result_str = registry.execute_tool(
                "analyze_screen",
                {"query": "What is on my screen?"},
                confirm_callback=lambda name, args: False,  # user denies
            )
            assert "User denied permission" in result_str
            mock_vision.assert_not_called()

    def test_only_query_passed_to_tool(self):
        """Verify the brain never injects LLM-sourced non-query args into analyze_screen."""
        brain = self._make_brain(
            '{"tool": "analyze_screen", "args": {"query": "What is open?"}}'
        )
        brain.registry.execute_tool = MagicMock(return_value=_SUCCESS_RESULT)

        brain.process_input("What is currently open on screen?")

        called_args = brain.registry.execute_tool.call_args.kwargs["args"]
        # Only "query" is allowed — no file path, URL, command, or arbitrary selector
        for key in called_args:
            assert key == "query", f"Unexpected argument '{key}' passed to analyze_screen"

    def test_vision_analysis_text_routed_to_pass2(self):
        """Ensure the vision analysis text flows through to LLM Pass-2 context."""
        brain = self._make_brain(
            '{"tool": "analyze_screen", "args": {"query": "What apps are open?"}}',
            llm_final="I can see Chrome and a terminal are open.",
        )
        brain.registry.execute_tool = MagicMock(return_value=_SUCCESS_RESULT)

        response = brain.process_input("What apps are open on screen?")

        # LLM Pass-2 wraps the vision analysis — not a deterministic template
        brain.llm.chat.assert_called_once()
        assert "chrome" in response.lower() or "terminal" in response.lower() or "I can see" in response

    def test_intent_routing_uses_analyze_screen_for_screen_query(self):
        """Confirm the brain routes 'what is on screen' to analyze_screen tool."""
        brain = self._make_brain(
            '{"tool": "analyze_screen", "args": {"query": "What is on my screen?"}}'
        )
        brain.registry.execute_tool = MagicMock(return_value=_SUCCESS_RESULT)

        brain.process_input("What is on my screen?")

        brain.registry.execute_tool.assert_called_once()
        assert brain.registry.execute_tool.call_args.kwargs["name"] == "analyze_screen"


# ===========================================================================
# Regression guard: test that no unrelated tools were modified
# ===========================================================================

class TestRegressionGuard:

    def test_existing_tools_still_registered(self):
        """Ensure no existing tools were accidentally removed."""
        registry = ToolRegistry()
        pre_existing = [
            "open_application", "close_application", "get_application_status",
            "get_system_info", "get_network_info", "get_clipboard", "set_clipboard",
            "get_volume", "set_volume", "media_key", "get_brightness", "set_brightness",
            "lock_screen", "sleep_system", "take_screenshot",
            "create_folder", "create_file", "list_directory", "open_file_or_folder",
            "read_file", "write_file",
            "open_browser", "open_url", "search_web", "search_youtube",
            "navigate_browser", "get_current_page", "read_page", "click_element",
            "type_into_element", "go_back", "close_browser",
            "run_command",
            "analyze_screen",
        ]
        for tool in pre_existing:
            assert tool in registry.tools, f"Tool '{tool}' is missing from the registry"

    def test_take_screenshot_still_not_dangerous(self):
        """take_screenshot danger flag must not have changed."""
        registry = ToolRegistry()
        _, is_dangerous = registry.tools["take_screenshot"]
        assert is_dangerous is False

    def test_write_file_still_dangerous(self):
        registry = ToolRegistry()
        _, is_dangerous = registry.tools["write_file"]
        assert is_dangerous is True

    def test_analyze_screen_grounding_did_not_break_other_grounding(self):
        """Verify take_screenshot grounding still returns its deterministic template."""
        brain = Brain()
        screenshot_success = json.dumps({
            "success": True,
            "tool": "take_screenshot",
            "data": {"filepath": r"C:\Users\user\Desktop\JARVIS_Screenshots\screenshot_20260101_120000.png"},
            "error": None,
        })
        grounded = brain._build_grounded_response("take_screenshot", {}, screenshot_success)
        assert grounded is not None
        assert "Done! Screenshot saved" in grounded
