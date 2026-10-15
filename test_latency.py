import pytest
from unittest.mock import patch, MagicMock
from core.brain import Brain
import json

@pytest.fixture
def brain():
    b = Brain()
    # Mock Pass-2 and execution to keep tests fast
    b.llm.chat = MagicMock(return_value="Mocked LLM Response")
    b.registry.execute_tool = MagicMock(return_value='{"success": True, "data": "Mocked"}')
    # Prevent test turns from polluting ~/.jarvis/history.json.
    # Same pattern used by test_phase27.py, test_phase28.py, test_phase29.py.
    with patch("core.session.SessionStore.save"):
        yield b


def test_fast_path_bypasses_ollama(brain):
    """Prove that fast-path matches do not call chat_json on Ollama."""
    with patch.object(brain.llm, 'chat_json') as mock_chat_json:
        # 1. Date/Time
        brain.process_input("What time is it?")
        mock_chat_json.assert_not_called()
        assert brain.registry.execute_tool.call_args[1]['name'] == 'get_datetime'

        # 2. Volume
        brain.process_input("Set volume to 50")
        mock_chat_json.assert_not_called()
        assert brain.registry.execute_tool.call_args[1]['name'] == 'set_volume'
        assert brain.registry.execute_tool.call_args[1]['args'] == {'level': 50}

        # 3. Brightness
        brain.process_input("Turn brightness to 75")
        mock_chat_json.assert_not_called()
        assert brain.registry.execute_tool.call_args[1]['name'] == 'set_brightness'
        assert brain.registry.execute_tool.call_args[1]['args'] == {'level': 75}

        # 4. YouTube override
        brain.process_input("Search YouTube for cat videos")
        mock_chat_json.assert_not_called()
        assert brain.registry.execute_tool.call_args[1]['name'] == 'search_youtube'
        assert brain.registry.execute_tool.call_args[1]['args'] == {'query': 'cat videos'}

def test_ambiguous_request_calls_ollama(brain):
    """Prove that non-fast-path requests that are not conversational still call the LLM."""
    # "Is Notepad running?" ends with '?' but is NOT matched by the positive/greeting
    # pre-classifiers → _is_skip stays False → chat_json is called for Phase 28 routing.
    with patch.object(brain.llm, 'chat_json', return_value='{"tool": null, "args": null}') as mock_chat_json:
        brain.process_input("Is Notepad running?")
        mock_chat_json.assert_called_once()

def test_chat_options_use_correct_latency_settings():
    """Verify that Pass-2 chat() uses the optimized latency options, and Pass-1 chat_json() remains unchanged."""
    from core.llm_client import LLMClient
    llm = LLMClient()
    with patch.object(llm.client, 'chat', return_value={'message': {'content': 'Mock'}}) as mock_ollama_chat:
        # 1. Verify Pass-2 chat() options
        llm.chat([{"role": "user", "content": "Hello"}])
        mock_ollama_chat.assert_called_with(
            model=llm.model,
            messages=[{"role": "user", "content": "Hello"}],
            options={"num_ctx": 4096, "num_predict": 200}
        )
        
        mock_ollama_chat.reset_mock()
        
        # 2. Verify Pass-1 chat_json() options (MUST remain 8192 for the large tool prompt)
        llm.chat_json([{"role": "user", "content": "Hello"}])
        mock_ollama_chat.assert_called_with(
            model=llm.model,
            messages=[{"role": "user", "content": "Hello"}],
            format="json",
            options={"num_predict": 80, "num_ctx": 8192}
        )



