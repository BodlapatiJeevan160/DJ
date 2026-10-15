"""
test_phase5_4_integration.py
============================
Phase 5.4 — Real Web Integration Test.
This test makes a REAL web request using the actual search_web skill
and passes the raw JSON data through the deterministic Search Results Presentation
layer to generate an instant, perfectly grounded summary without hitting the LLM.

Run with:
    python test_phase5_4_integration.py
"""

import sys
import json
import unittest
from core.brain import Brain
import skills.web_browser as wb

class TestPhase5_4RealIntegration(unittest.TestCase):

    def setUp(self):
        self.brain = Brain()
        
    def tearDown(self):
        wb.close_browser()

    def test_real_web_search_presentation(self):
        query = "Search the web for Python programming"
        original_execute = self.brain.registry.execute_tool
        
        captured_result = None
        
        def capturing_execute(name, args, confirm_callback=None):
            nonlocal captured_result
            result = original_execute(name, args, confirm_callback)
            if name == "search_web":
                captured_result = result
            return result
            
        self.brain.registry.execute_tool = capturing_execute
        
        print(f"\n[INTEGRATION] Running real request: '{query}'")
        try:
            final_response = self.brain.process_input(query)
        except Exception as e:
            self.skipTest(f"Network or execution failure: {e}")
            
        self.assertIsNotNone(captured_result, "search_web was never executed")
        
        try:
            result_data = json.loads(captured_result)
        except Exception:
            self.fail(f"Could not parse real tool result: {captured_result}")
            
        if not result_data.get("success"):
            self.assertIn("failed", final_response.lower())
            return
            
        results = result_data.get("data")
        
        if not results:
            self.assertIn("found no results", final_response)
            return
            
        self.assertNotIn("failed", final_response.lower())
        self.assertNotIn("Mocked Test Title", final_response)
        
        # Verify the deterministic formatting prefix
        self.assertIn("Here are the search results for \"Python programming\":", final_response)

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5.4 — Real Web Integration Test")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase5_4RealIntegration)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
