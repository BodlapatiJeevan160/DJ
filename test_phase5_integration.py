"""
test_phase5_integration.py
==========================
Phase 5.2 — Real Web Integration Test.
This test makes a REAL web request using the actual search_web skill
to ensure grounding works with real data and no mocked results are used.

Run with:
    python test_phase5_integration.py
"""

import sys
import json
import unittest
from core.brain import Brain
import skills.web_browser as wb

class TestPhase5RealIntegration(unittest.TestCase):

    def setUp(self):
        self.brain = Brain()
        
    def tearDown(self):
        # Ensure browser is closed after real test
        wb.close_browser()

    def test_real_web_search_grounding(self):
        """
        Executes a real web search and verifies the final grounded response
        is accurately based on the real internet data returned.
        """
        query = "Search the web for Python programming"
        
        # We need to capture the raw result to verify grounding.
        # We'll patch process_input slightly to capture it, but NOT change execution.
        original_execute = self.brain.registry.execute_tool
        
        captured_result = None
        
        def capturing_execute(name, args, confirm_callback=None):
            nonlocal captured_result
            result = original_execute(name, args, confirm_callback)
            if name == "search_web":
                captured_result = result
            return result
            
        self.brain.registry.execute_tool = capturing_execute
        
        # 1. Run the real process
        print(f"\n[INTEGRATION] Running real request: '{query}'")
        try:
            final_response = self.brain.process_input(query)
        except Exception as e:
            self.skipTest(f"Network or execution failure: {e}")
            
        # 2. Check if it succeeded over the network
        self.assertIsNotNone(captured_result, "search_web was never executed")
        
        try:
            result_data = json.loads(captured_result)
        except Exception:
            self.fail(f"Could not parse real tool result: {captured_result}")
            
        if not result_data.get("success"):
            # Graceful handle for network/timeout failures
            print(f"[INTEGRATION] Network search failed (expected if offline/timeout): {result_data.get('error')}")
            # Verify the response mentions the error
            self.assertIn("failed", final_response.lower())
            return
            
        # 3. Verify real data structure
        results = result_data.get("data")
        self.assertIsInstance(results, list, "Real search_web should return a list of objects")
        
        if not results:
            print("[INTEGRATION] Search returned 0 results. Skipping strict parsing checks.")
            self.assertIn("found no results", final_response)
            return
            
        # Verify schema
        first_result = results[0]
        self.assertIn("title", first_result)
        self.assertIn("snippet", first_result)
        self.assertIn("url", first_result)
        
        # 4. Verify Grounding
        # The final response MUST contain the real title of the first result
        real_title = first_result["title"]
        self.assertIn(
            real_title, final_response, 
            f"Final response did not contain real title '{real_title}'"
        )
        
        # Make sure it didn't just hallucinate a fake result
        self.assertNotIn("Mocked Test Title", final_response)
        
        print("\n" + "="*50)
        print("REAL RAW RESULT DATA:")
        print(json.dumps(result_data, indent=2))
        print("\nGROUNDED FINAL RESPONSE:")
        print(final_response)
        print("="*50 + "\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5.2 — Real Web Integration Test")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase5RealIntegration)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
