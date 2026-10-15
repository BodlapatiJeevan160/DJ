import sys

lines = open('test_phase26.py').readlines()

# We want to keep everything up to line 248.
with open("test_phase26.py", "w") as f:
    f.writelines(lines[:248])
    f.write(r"            r'|'" + "\n")
    f.write(r"            r'\b(search\s+for|find)\s+[\'\"].+?[\'\"]'" + "\n")
    f.write(r"            r'|'" + "\n")
    f.write(r"            r'^grep\s+'" + "\n")
    f.write(r"            r'|'" + "\n")
    f.write(r"            r'\bmy\s+(project|workspace|codebase|repo(?:sitory)?|code|files?|source)\b'," + "\n")
    f.write(r"            re.IGNORECASE" + "\n")
    f.write(r"        )" + "\n")
    f.write(r"        if tool_name in _UNREGISTERED_SEARCH_ALIASES and _WORKSPACE_SEARCH_INTENT.search(user_input):" + "\n")
    
    # In my last fix script I started at line 359, which happened to be inside the if-block. Let's make sure we find the right line to resume.
    # We want to resume right after `if tool_name in _UNREGISTERED_SEARCH_ALIASES ...:`
    
    # The original file's if-block starts with `tool_name = "search_workspace"`
    # Let's find that line
    start_idx = -1
    for i, line in enumerate(lines):
        if 'tool_name = "search_workspace"' in line and 'clean =' in lines[i+2]:
            start_idx = i
            break
            
    f.writelines(lines[start_idx:start_idx+93])
    
    f.write("""
class TestPhase26GuardF(unittest.TestCase):
    \"\"\"Regression tests for Guard F: Prior result reference routing.\"\"\"
    def setUp(self):
        from core.brain import Brain
        self.brain = Brain()
        
        # We replace the actual execution with a dummy return so we don't really touch disk
        self.brain.registry.execute_tool = MagicMock(return_value=json.dumps({
            "success": True, 
            "data": {"results": [{"file": "dummy.txt", "match": "MATCH"}]}
        }))
        
        # Default mock for LLM text gen
        self.brain.llm.chat = MagicMock(return_value="Pass 2 generated text.")

    def test_guard_f_suppresses_unrelated_tool(self):
        # Setup: populate history with a previous search
        self.brain.history.append({
            "role": "system",
            "content": "ACTION EXECUTED: search_workspace\\nRESULT: {}"
        })
        
        # User references it
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "get_network_info", "args": {}}')
        response = self.brain.process_input("Summarize the first result.")
        
        # Tool execution should be skipped
        self.brain.registry.execute_tool.assert_not_called()
        self.assertEqual(response, "Pass 2 generated text.")

    def test_guard_f_allows_navigation_tool(self):
        # Setup: populate history with a previous search
        self.brain.history.append({
            "role": "system",
            "content": "ACTION EXECUTED: search_workspace\\nRESULT: {}"
        })
        
        # User references it, but LLM picks a VALID navigation tool (e.g. edit_file)
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "edit_file", "args": {"path": "x"}}')
        
        # Reset execution mock to return dummy JSON
        self.brain.registry.execute_tool.reset_mock()
        self.brain.registry.execute_tool.return_value = '{"success": true}'
        
        self.brain.process_input("Open the first result.")
        
        # Should have executed edit_file
        self.brain.registry.execute_tool.assert_called_once()
        args, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs["name"], "edit_file")

    def test_guard_f_clarification_if_no_history(self):
        # History is empty
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "get_network_info", "args": {}}')
        response = self.brain.process_input("Summarize the first result.")
        
        # Tool execution skipped, clarification returned
        self.brain.registry.execute_tool.assert_not_called()
        self.assertIn("I don't have a previous search result to summarize", response)

    def test_independent_tool_not_suppressed(self):
        # History has previous search
        self.brain.history.append({
            "role": "system",
            "content": "ACTION EXECUTED: search_workspace\\nRESULT: {}"
        })
        
        # User asks unrelated question NOT matching pattern
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "get_network_info", "args": {}}')
        
        self.brain.process_input("What is my IP address?")
        
        # Tool execution SHOULD occur
        self.brain.registry.execute_tool.assert_called_once()
        args, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs["name"], "get_network_info")

if __name__ == '__main__':
    unittest.main()
""")
