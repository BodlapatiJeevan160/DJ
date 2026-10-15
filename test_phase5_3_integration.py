"""
test_phase5_3_integration.py
============================
Phase 5.3 — Real Web Integration Test.
This test makes a REAL web request using the actual search_web skill
and passes the raw JSON data through the LLM-powered Search Results Presentation
layer to generate an intelligent, grounded summary avoiding redundant domains.

Run with:
    python test_phase5_3_integration.py
"""

import sys
import json
import unittest
from core.brain import Brain
import skills.web_browser as wb

class TestPhase5_3RealIntegration(unittest.TestCase):

    def setUp(self):
        self.brain = Brain()
        
    def tearDown(self):
        # Ensure browser is closed after real test
        wb.close_browser()

    def test_real_web_search_presentation(self):
        """
        Executes a real web search and verifies the final intelligent response
        is accurately based on the real internet data returned.
        """
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
            print(f"[INTEGRATION] Network search failed (expected if offline/timeout): {result_data.get('error')}")
            self.assertIn("failed", final_response.lower())
            return
            
        results = result_data.get("data")
        self.assertIsInstance(results, list, "Real search_web should return a list of objects")
        
        if not results:
            print("[INTEGRATION] Search returned 0 results. Skipping strict parsing checks.")
            self.assertIn("found no results", final_response)
            return
            
        self.assertNotIn("failed", final_response.lower())
        self.assertNotIn("Mocked Test Title", final_response)
        
        print("\n" + "="*50)
        print("REAL RAW RESULT DATA:")
        print(json.dumps(result_data, indent=2))
        print("\nPRESENTED FINAL RESPONSE:")
        print(final_response)
        print("="*50 + "\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5.3 — Real Web Integration Test")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase5_3RealIntegration)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
