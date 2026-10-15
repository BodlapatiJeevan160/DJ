"""
test_phase10.py
Phase 10 -- Secure Volume and Media Control Tests.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from skills import system_ops
from core.tool_registry import ToolRegistry
from core.brain import Brain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vol_result(level):
    return json.dumps({"success": True, "tool": "get_volume",
                       "data": {"level": level}, "error": None})

def _set_vol_result(level):
    return json.dumps({"success": True, "tool": "set_volume",
                       "data": {"level": level}, "error": None})

def _media_result(action):
    return json.dumps({"success": True, "tool": "media_key",
                       "data": {"action": action}, "error": None})

def _failure_result(tool, error):
    return json.dumps({"success": False, "tool": tool,
                       "data": None, "error": error})


# ---------------------------------------------------------------------------
# get_volume unit tests (1-4)
# ---------------------------------------------------------------------------

class TestGetVolume:

    def _mock_pycaw(self, scalar):
        mock_vol = MagicMock()
        mock_vol.GetMasterVolumeLevelScalar.return_value = scalar
        mock_interface = MagicMock()
        mock_interface.QueryInterface.return_value = mock_vol
        mock_devices = MagicMock()
        mock_devices.Activate.return_value = mock_interface
        return patch.dict("sys.modules", {
            "pycaw": MagicMock(),
            "pycaw.pycaw": MagicMock(
                AudioUtilities=MagicMock(GetSpeakers=MagicMock(return_value=mock_devices)),
                IAudioEndpointVolume=MagicMock(_iid_="iid"),
            ),
            "comtypes": MagicMock(CLSCTX_ALL=1),
        })

    def test_get_volume_success(self):
        with self._mock_pycaw(0.65):
            res = system_ops.get_volume()
        assert res["success"] is True
        assert res["tool"] == "get_volume"
        assert res["data"]["level"] == 65
        assert res["error"] is None

    def test_get_volume_boundary_zero(self):
        with self._mock_pycaw(0.0):
            res = system_ops.get_volume()
        assert res["success"] is True
        assert res["data"]["level"] == 0

    def test_get_volume_no_device(self):
        with patch.dict("sys.modules", {
            "pycaw": MagicMock(),
            "pycaw.pycaw": MagicMock(
                AudioUtilities=MagicMock(GetSpeakers=MagicMock(
                    side_effect=Exception("No audio device"))),
                IAudioEndpointVolume=MagicMock(_iid_="iid"),
            ),
            "comtypes": MagicMock(CLSCTX_ALL=1),
        }):
            res = system_ops.get_volume()
        assert res["success"] is False
        assert "No audio device" in res["error"]

    def test_get_volume_import_error(self):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if "pycaw" in name:
                raise ImportError("No module named pycaw")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            res = system_ops.get_volume()
        assert res["success"] is False
        assert res["error"] is not None


# ---------------------------------------------------------------------------
# set_volume unit tests (5-15)
# ---------------------------------------------------------------------------

class TestSetVolume:

    def _mock_pycaw(self):
        mock_vol = MagicMock()
        mock_interface = MagicMock()
        mock_interface.QueryInterface.return_value = mock_vol
        mock_devices = MagicMock()
        mock_devices.Activate.return_value = mock_interface
        return patch.dict("sys.modules", {
            "pycaw": MagicMock(),
            "pycaw.pycaw": MagicMock(
                AudioUtilities=MagicMock(GetSpeakers=MagicMock(return_value=mock_devices)),
                IAudioEndpointVolume=MagicMock(_iid_="iid"),
            ),
            "comtypes": MagicMock(CLSCTX_ALL=1),
        })

    def test_set_volume_success_50(self):
        with self._mock_pycaw():
            res = system_ops.set_volume(50)
        assert res["success"] is True
        assert res["tool"] == "set_volume"
        assert res["data"]["level"] == 50

    def test_set_volume_success_0(self):
        with self._mock_pycaw():
            res = system_ops.set_volume(0)
        assert res["success"] is True
        assert res["data"]["level"] == 0

    def test_set_volume_success_100(self):
        with self._mock_pycaw():
            res = system_ops.set_volume(100)
        assert res["success"] is True
        assert res["data"]["level"] == 100

    def test_set_volume_negative(self):
        res = system_ops.set_volume(-1)
        assert res["success"] is False
        assert "between 0 and 100" in res["error"]

    def test_set_volume_over_100(self):
        res = system_ops.set_volume(101)
        assert res["success"] is False
        assert "between 0 and 100" in res["error"]

    def test_set_volume_float(self):
        res = system_ops.set_volume(70.5)
        assert res["success"] is False
        assert "integer" in res["error"].lower()

    def test_set_volume_string(self):
        res = system_ops.set_volume("70")
        assert res["success"] is False

    def test_set_volume_none(self):
        res = system_ops.set_volume(None)
        assert res["success"] is False

    def test_set_volume_bool_rejected(self):
        assert system_ops.set_volume(True)["success"]  is False
        assert system_ops.set_volume(False)["success"] is False

    def test_set_volume_exception(self):
        mock_vol = MagicMock()
        mock_vol.SetMasterVolumeLevelScalar.side_effect = Exception("COM error")
        mock_interface = MagicMock()
        mock_interface.QueryInterface.return_value = mock_vol
        mock_devices = MagicMock()
        mock_devices.Activate.return_value = mock_interface
        with patch.dict("sys.modules", {
            "pycaw": MagicMock(),
            "pycaw.pycaw": MagicMock(
                AudioUtilities=MagicMock(GetSpeakers=MagicMock(return_value=mock_devices)),
                IAudioEndpointVolume=MagicMock(_iid_="iid"),
            ),
            "comtypes": MagicMock(CLSCTX_ALL=1),
        }):
            res = system_ops.set_volume(70)
        assert res["success"] is False
        assert "COM error" in res["error"]


# ---------------------------------------------------------------------------
# media_key unit tests (16-29)
# ---------------------------------------------------------------------------

class TestMediaKey:

    def _call(self, action):
        with patch("skills.system_ops.pyautogui") as mock_pg:
            res = system_ops.media_key(action)
            return res, mock_pg

    def test_media_key_play_pause(self):
        res, mock_pg = self._call("play_pause")
        assert res["success"] is True
        mock_pg.press.assert_called_once_with("playpause")

    def test_media_key_next_track(self):
        res, mock_pg = self._call("next_track")
        assert res["success"] is True
        mock_pg.press.assert_called_once_with("nexttrack")

    def test_media_key_prev_track(self):
        res, mock_pg = self._call("prev_track")
        assert res["success"] is True
        mock_pg.press.assert_called_once_with("prevtrack")

    def test_media_key_mute(self):
        res, mock_pg = self._call("mute")
        assert res["success"] is True
        mock_pg.press.assert_called_once_with("volumemute")

    def test_media_key_volume_up(self):
        res, mock_pg = self._call("volume_up")
        assert res["success"] is True
        mock_pg.press.assert_called_once_with("volumeup")

    def test_media_key_volume_down(self):
        res, mock_pg = self._call("volume_down")
        assert res["success"] is True
        mock_pg.press.assert_called_once_with("volumedown")

    def test_media_key_invalid_action(self):
        res, mock_pg = self._call("stop")
        assert res["success"] is False
        assert "Invalid media action" in res["error"]
        mock_pg.press.assert_not_called()

    def test_media_key_empty_string(self):
        res, mock_pg = self._call("")
        assert res["success"] is False
        mock_pg.press.assert_not_called()

    def test_media_key_none(self):
        with patch("skills.system_ops.pyautogui") as mock_pg:
            res = system_ops.media_key(None)
        assert res["success"] is False
        mock_pg.press.assert_not_called()

    def test_media_key_injection(self):
        res, mock_pg = self._call("rm -rf /")
        assert res["success"] is False
        mock_pg.press.assert_not_called()

    def test_media_key_numeric(self):
        with patch("skills.system_ops.pyautogui") as mock_pg:
            res = system_ops.media_key(42)
        assert res["success"] is False
        mock_pg.press.assert_not_called()

    def test_media_key_exception(self):
        with patch("skills.system_ops.pyautogui") as mock_pg:
            mock_pg.press.side_effect = Exception("display error")
            res = system_ops.media_key("mute")
        assert res["success"] is False
        assert "display error" in res["error"]

    def test_raw_action_never_passed_to_pyautogui(self):
        with patch("skills.system_ops.pyautogui") as mock_pg:
            system_ops.media_key("mute")
        call_args = mock_pg.press.call_args[0]
        assert call_args[0] == "volumemute"
        assert "mute" not in call_args


# ---------------------------------------------------------------------------
# Tool registry tests (30-32)
# ---------------------------------------------------------------------------

class TestToolRegistry:

    def test_volume_tools_registered(self):
        registry = ToolRegistry()
        assert "get_volume" in registry.tools
        assert "set_volume" in registry.tools
        assert "media_key"  in registry.tools

    def test_volume_tools_not_dangerous(self):
        registry = ToolRegistry()
        for name in ("get_volume", "set_volume", "media_key"):
            _, dangerous = registry.tools[name]
            assert dangerous is False

    def test_existing_tools_unchanged(self):
        registry = ToolRegistry()
        for tool in ("get_clipboard", "set_clipboard", "get_system_info",
                     "open_application", "close_application", "get_application_status",
                     "create_folder", "create_file", "list_directory",
                     "open_file_or_folder", "read_file",
                     "search_web", "search_youtube", "open_browser"):
            assert tool in registry.tools, f"Regression: {tool} missing"


# ---------------------------------------------------------------------------
# Brain grounding tests (33-43)
# ---------------------------------------------------------------------------

class TestBrainGrounding:

    def setup_method(self):
        self.brain = Brain()

    def test_grounding_get_volume_success_none(self):
        grounded = self.brain._build_grounded_response("get_volume", {}, _vol_result(65))
        assert grounded is None

    def test_grounding_get_volume_failure(self):
        grounded = self.brain._build_grounded_response(
            "get_volume", {}, _failure_result("get_volume", "No device"))
        assert grounded is not None
        assert "volume" in grounded.lower()
        assert "failed" in grounded.lower()

    def test_grounding_set_volume_success(self):
        grounded = self.brain._build_grounded_response(
            "set_volume", {"level": 70}, _set_vol_result(70))
        assert grounded is not None
        assert "70%" in grounded

    def test_grounding_set_volume_failure(self):
        grounded = self.brain._build_grounded_response(
            "set_volume", {"level": -1},
            _failure_result("set_volume", "Volume level must be an integer between 0 and 100."))
        assert grounded is not None
        assert "failed" in grounded.lower()

    def test_grounding_media_key_play_pause(self):
        assert self.brain._build_grounded_response(
            "media_key", {"action": "play_pause"}, _media_result("play_pause")) == "Toggled play/pause."

    def test_grounding_media_key_next_track(self):
        assert self.brain._build_grounded_response(
            "media_key", {"action": "next_track"}, _media_result("next_track")) == "Skipped to the next track."

    def test_grounding_media_key_prev_track(self):
        assert self.brain._build_grounded_response(
            "media_key", {"action": "prev_track"}, _media_result("prev_track")) == "Went back to the previous track."

    def test_grounding_media_key_mute(self):
        assert self.brain._build_grounded_response(
            "media_key", {"action": "mute"}, _media_result("mute")) == "Toggled mute."

    def test_grounding_media_key_volume_up(self):
        assert self.brain._build_grounded_response(
            "media_key", {"action": "volume_up"}, _media_result("volume_up")) == "Turned the volume up."

    def test_grounding_media_key_volume_down(self):
        assert self.brain._build_grounded_response(
            "media_key", {"action": "volume_down"}, _media_result("volume_down")) == "Turned the volume down."

    def test_grounding_media_key_failure(self):
        grounded = self.brain._build_grounded_response(
            "media_key", {"action": "stop"},
            _failure_result("media_key", "Invalid media action stop."))
        assert grounded is not None
        assert "failed" in grounded.lower()


# ---------------------------------------------------------------------------
# Integration tests (44-50)
# ---------------------------------------------------------------------------

class TestBrainIntegration:

    def _make_brain(self, intent_json, llm_final="Volume noted."):
        brain = Brain()
        brain.llm.chat_json = MagicMock(return_value=intent_json)
        brain.llm.chat      = MagicMock(return_value=llm_final)
        return brain

    def test_set_volume_pipeline(self):
        brain = self._make_brain('{"tool": "set_volume", "args": {"level": 70}}')
        brain.registry.execute_tool = MagicMock(return_value=_set_vol_result(70))
        response = brain.process_input("Set the volume to 70")
        brain.registry.execute_tool.assert_called_once_with(
            name="set_volume", args={"level": 70}, confirm_callback=brain._ask_confirmation)
        assert "70%" in response
        brain.llm.chat.assert_not_called()

    def test_get_volume_pipeline(self):
        brain = self._make_brain('{"tool": "get_volume", "args": {}}',
                                 llm_final="The current volume is 65%.")
        brain.registry.execute_tool = MagicMock(return_value=_vol_result(65))
        response = brain.process_input("What is the current volume?")
        brain.registry.execute_tool.assert_called_once_with(
            name="get_volume", args={}, confirm_callback=brain._ask_confirmation)
        brain.llm.chat.assert_called_once()
        assert "65%" in response

    def test_media_key_play_pause_pipeline(self):
        brain = self._make_brain('{"tool": "media_key", "args": {"action": "play_pause"}}')
        brain.registry.execute_tool = MagicMock(return_value=_media_result("play_pause"))
        response = brain.process_input("Pause the music")
        brain.registry.execute_tool.assert_called_once_with(
            name="media_key", args={"action": "play_pause"}, confirm_callback=brain._ask_confirmation)
        assert response == "Toggled play/pause."
        brain.llm.chat.assert_not_called()

    def test_media_key_next_track_pipeline(self):
        brain = self._make_brain('{"tool": "media_key", "args": {"action": "next_track"}}')
        brain.registry.execute_tool = MagicMock(return_value=_media_result("next_track"))
        response = brain.process_input("Skip this track")
        assert response == "Skipped to the next track."
        brain.llm.chat.assert_not_called()

    def test_media_key_mute_pipeline(self):
        brain = self._make_brain('{"tool": "media_key", "args": {"action": "mute"}}')
        brain.registry.execute_tool = MagicMock(return_value=_media_result("mute"))
        response = brain.process_input("Mute please")
        assert response == "Toggled mute."

    def test_set_volume_failure_pipeline(self):
        brain = self._make_brain('{"tool": "set_volume", "args": {"level": -1}}')
        brain.registry.execute_tool = MagicMock(
            return_value=_failure_result("set_volume",
                "Volume level must be an integer between 0 and 100."))
        response = brain.process_input("Set volume to negative 1")
        assert "failed" in response.lower()
        brain.llm.chat.assert_not_called()

    def test_mute_correction_guard(self):
        brain = self._make_brain('{"tool": "open_application", "args": {"application": "mute"}}')
        captured = {}
        def fake_execute(name, args, confirm_callback=None):
            captured["name"] = name
            captured["args"] = args
            return _media_result("mute")
        brain.registry.execute_tool = fake_execute
        response = brain.process_input("Mute")
        assert captured.get("name") == "media_key", f"Expected media_key, got {captured.get('name')}"
        assert captured.get("args") == {"action": "mute"}
        assert response == "Toggled mute."


# ---------------------------------------------------------------------------
# Phase 9 regression
# ---------------------------------------------------------------------------

class TestPhase9Regression:

    def test_get_clipboard_still_routes_to_llm(self):
        brain = Brain()
        result = json.dumps({"success": True, "tool": "get_clipboard",
                             "data": {"text": "Hello"}, "error": None})
        assert brain._build_grounded_response("get_clipboard", {}, result) is None

    def test_set_clipboard_still_deterministic(self):
        brain = Brain()
        result = json.dumps({"success": True, "tool": "set_clipboard",
                             "data": {"written": 5}, "error": None})
        grounded = brain._build_grounded_response("set_clipboard", {"text": "hello"}, result)
        assert grounded is not None
        assert "clipboard" in grounded.lower()
        assert "5" in grounded

    def test_clipboard_registry_unchanged(self):
        registry = ToolRegistry()
        _, get_d = registry.tools["get_clipboard"]
        _, set_d = registry.tools["set_clipboard"]
        assert get_d is False
        assert set_d is False
