import sys
from unittest.mock import patch
from core.brain import Brain

@patch("builtins.input", return_value="y")
@patch("core.llm_client.LLMClient.chat_json")
@patch("core.llm_client.LLMClient.chat")
def test_jarvis(mock_chat, mock_chat_json, mock_input):
    print("Initializing Brain for testing...")
    brain = Brain()
    
    mock_chat_json.side_effect = [
        '{"tool": "open_application", "args": {"application": "Notepad"}}',
        '{"tool": "create_folder", "args": {"path": "JarvisTest"}}',
        '{"tool": "get_system_info", "args": {}}',
        '{"tool": "run_command", "args": {"command": "rmdir JarvisTest"}}',
        '{"tool": "None", "args": {}}'
    ]
    mock_chat.return_value = "I don't have a tool to send emails yet."

    print("\n" + "="*50)
    print("TEST 1: Open Notepad")
    response = brain.process_input("Open Notepad.")
    print(f"FINAL RESPONSE: {response}")

    print("\n" + "="*50)
    print("TEST 2: Create a folder called JarvisTest on my Desktop")
    response = brain.process_input("Create a folder called JarvisTest on my Desktop.")
    print(f"FINAL RESPONSE: {response}")
    
    print("\n" + "="*50)
    print("TEST 3: What is my computer's operating system?")
    response = brain.process_input("What is my computer's operating system?")
    print(f"FINAL RESPONSE: {response}")

    print("\n" + "="*50)
    print("TEST 4: Delete JarvisTest (Requires confirmation)")
    with patch('builtins.input', return_value='n'):
        response = brain.process_input("Delete JarvisTest on my Desktop.")
        print(f"FINAL RESPONSE: {response}")

    print("\n" + "="*50)
    print("TEST 5: Send an email to someone")
    response = brain.process_input("Send an email to someone.")
    print(f"FINAL RESPONSE: {response}")

if __name__ == "__main__":
    test_jarvis()
