"""
test_phase20.py
===============
Phase 20 -- Text-to-Speech (TTS) Tests

Verifies:
  - AudioEngine lifecycle (start, stop, disabled-by-default)
  - Tool interfaces (toggle_speech, set_voice_parameters)
  - Sanitization (Markdown, code, URLs, control characters, max length limit)
  - Queue ordering and non-blocking background behavior
  - Parameter validation and bounds
  - Handling of missing audio drivers or pyttsx3 initialization failures
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from core.audio import AudioEngine
from skills.audio_ops import toggle_speech, set_voice_parameters
from core.tool_registry import ToolRegistry

# ===========================================================================
# Unit Tests: core/audio.py
# ===========================================================================

class TestAudioEngineLifecycle:
    @patch("pyttsx3.init")
    def test_disabled_by_default(self, mock_init):
        engine = AudioEngine()
        assert engine.enabled is False
        
    @patch("pyttsx3.init")
    def test_enable_disable(self, mock_init):
        engine = AudioEngine()
        engine.start()
        engine.toggle(True)
        assert engine.enabled is True
        engine.toggle(False)
        assert engine.enabled is False
        engine.stop()

    @patch("pyttsx3.init")
    def test_missing_driver_fallback(self, mock_init):
        # Simulate pyttsx3 failing to initialize
        mock_init.side_effect = Exception("No audio driver found")
        engine = AudioEngine()
        engine.start()
        
        # Wait a moment for worker to fail and exit
        time.sleep(0.2)
        assert engine.running is False
        assert engine.init_error == "No audio driver found"
        
        # Toggle should raise
        with pytest.raises(RuntimeError) as exc:
            engine.toggle(True)
        assert "failed to initialize" in str(exc.value)
        
        # Should not crash main thread when trying to speak (though toggle prevents it anyway)
        engine.speak("Test")

    @patch("pyttsx3.init")
    def test_clean_shutdown(self, mock_init):
        engine = AudioEngine()
        engine.start()
        assert engine.running is True
        engine.stop()
        assert engine.running is False
        assert engine.worker_thread is not None
        assert not engine.worker_thread.is_alive()


class TestAudioEngineSanitization:
    def test_strip_markdown_and_code(self):
        engine = AudioEngine()
        raw = "Here is some code: ```python\nprint('hello')\n``` and `inline`."
        sanitized = engine._sanitize(raw)
        assert "```" not in sanitized
        assert "`inline`" not in sanitized
        assert "Here is some code" in sanitized
        
    def test_strip_urls(self):
        engine = AudioEngine()
        raw = "Visit https://google.com for more info."
        sanitized = engine._sanitize(raw)
        assert "https://google.com" not in sanitized
        assert "Visit" in sanitized
        
    def test_strip_control_characters(self):
        engine = AudioEngine()
        raw = "Hello\x00World\x1F!"
        sanitized = engine._sanitize(raw)
        assert "\x00" not in sanitized
        assert "Hello World !" in sanitized
        
    @patch("pyttsx3.init")
    def test_max_character_limit(self, mock_init):
        mock_pyttsx3 = MagicMock()
        mock_init.return_value = mock_pyttsx3
        
        engine = AudioEngine()
        engine.start()
        raw = "A" * 600
        engine.toggle(True)
        
        engine.speak(raw)
        # Wait a moment for worker to process
        time.sleep(0.2)
        
        calls = mock_pyttsx3.say.call_args_list
        assert len(calls) == 1
        item = calls[0][0][0]
        assert len(item) <= 500
        assert item.endswith("...")
        engine.stop()

    @patch("pyttsx3.init")
    def test_ignore_json_and_debug(self, mock_init):
        mock_pyttsx3 = MagicMock()
        mock_init.return_value = mock_pyttsx3
        
        engine = AudioEngine()
        engine.start()
        engine.toggle(True)
        
        # Should be rejected
        engine.speak('{"success": true, "data": {}}')
        engine.speak("Traceback (most recent call last)")
        
        time.sleep(0.2)
        mock_pyttsx3.say.assert_not_called()
        engine.stop()


class TestAudioEngineConcurrency:
    @patch("pyttsx3.init")
    def test_queue_ordering_and_non_blocking(self, mock_init):
        # Create a mock engine that takes time to speak
        mock_pyttsx3 = MagicMock()
        
        # simulate say taking 100ms
        def fake_runAndWait():
            time.sleep(0.1)
            
        mock_pyttsx3.runAndWait = fake_runAndWait
        mock_init.return_value = mock_pyttsx3
        
        engine = AudioEngine()
        engine.start()
        engine.toggle(True)
        
        start_time = time.time()
        # Enqueue multiple items rapidly
        engine.speak("One")
        engine.speak("Two")
        engine.speak("Three")
        end_time = time.time()
        
        # Verify it was non-blocking (took less than 0.1s to enqueue 3 items)
        assert (end_time - start_time) < 0.1
        
        # Wait for worker to process all
        time.sleep(0.5)
        engine.stop()
        
        # Verify order of calls
        calls = [c[0][0] for c in mock_pyttsx3.say.call_args_list]
        assert calls == ["One", "Two", "Three"]


# ===========================================================================
# Unit Tests: skills/audio_ops.py
# ===========================================================================

class TestAudioOpsTools:
    @patch("skills.audio_ops.engine")
    def test_toggle_speech_validation(self, mock_engine):
        mock_engine.toggle.side_effect = lambda state: state
        
        res = toggle_speech("yes")
        assert res["success"] is False
        assert "boolean" in res["error"]
        
        res = toggle_speech(True)
        assert res["success"] is True
        assert res["data"]["state"] == "enabled"
        
        res = toggle_speech(False)
        assert res["success"] is True
        assert res["data"]["state"] == "disabled"
        
    @patch("skills.audio_ops.engine")
    def test_toggle_speech_init_failure(self, mock_engine):
        mock_engine.toggle.side_effect = RuntimeError("TTS engine failed to initialize")
        res = toggle_speech(True)
        assert res["success"] is False
        assert "failed to initialize" in res["error"]

    def test_set_voice_parameters_validation(self):
        # Rate validation
        res = set_voice_parameters(rate=999)
        assert res["success"] is False
        
        res = set_voice_parameters(rate="fast")
        assert res["success"] is False
        
        res = set_voice_parameters(rate=250)
        assert res["success"] is True
        
        # Volume validation
        res = set_voice_parameters(volume=1.5)
        assert res["success"] is False
        
        res = set_voice_parameters(volume=-0.1)
        assert res["success"] is False
        
        res = set_voice_parameters(volume=0.8)
        assert res["success"] is True
        
        # Voice Index validation
        res = set_voice_parameters(voice_index=-1)
        assert res["success"] is False
        
        res = set_voice_parameters(voice_index=1)
        assert res["success"] is True


# ===========================================================================
# Tool Registry Integration
# ===========================================================================

class TestToolRegistryPhase20:
    def test_tools_registered(self):
        registry = ToolRegistry()
        assert "toggle_speech" in registry.tools
        assert "set_voice_parameters" in registry.tools
        
        # Tools should not be dangerous
        assert registry.tools["toggle_speech"][1] is False
        assert registry.tools["set_voice_parameters"][1] is False

    def test_existing_functionality_unchanged(self):
        registry = ToolRegistry()
        # 41(p19) + 2(p20) = 43 tools
        assert len(registry.tools) >= 43
        assert "remember" in registry.tools
        assert "open_browser" in registry.tools

class TestBrainTTSIntegration:
    def test_brain_intent_prompt_prevents_speech_hallucination(self):
        from core.brain import Brain
        brain = Brain()

        # "Adjust the speech volume?" has no fast-path, ends with '?' so
        # _is_skip stays False, and MEDIA_AUDIO is selected by Phase 28
        # (both "speech" and "volume" score 3) → toggle_speech and
        # set_voice_parameters appear in the intent_system_prompt.
        brain.llm.chat_json = MagicMock(return_value='{"tool": null, "args": null}')
        brain.llm.chat = MagicMock(return_value="I can adjust the speech volume for you.")

        brain.process_input("Adjust the speech volume?")

        # chat_json must have been called for the assertions below to be valid.
        assert brain.llm.chat_json.call_count == 1, (
            "chat_json was not called — the phrase must reach Pass-1 without fast-pathing."
        )
        messages = brain.llm.chat_json.call_args[0][0]
        sys_prompt = messages[0]["content"]

        assert "NEVER attempt to call a tool named 'speech', 'say', or 'speak'" in sys_prompt
        assert "toggle_speech" in sys_prompt
        assert "set_voice_parameters" in sys_prompt

