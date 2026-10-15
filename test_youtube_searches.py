"""
Phase 4 YouTube search tests.
Runs all 5 test cases sequentially in ONE brain session.
"""
from core.brain import Brain

brain = Brain()

tests = [
    ("TEST 1: Search YouTube for Blender tutorials",    "Search YouTube for Blender tutorials"),
    ("TEST 2: Search YouTube for Python tutorials",     "Search YouTube for Python tutorials"),
    ("TEST 3: Search the web for Python programming",   "Search the web for Python programming"),
    ("TEST 4: Search the web for the latest AI news",   "Search the web for the latest AI news"),
]

for label, query in tests:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    brain.process_input(query)

print("\n" + "="*60)
print("TEST 5: YouTube failure simulation (invalid URL to trigger error path)")
print("="*60)
# Temporarily monkey-patch the page.goto to raise so we can verify error reporting
import skills.web_browser as wb
original_page = wb._page

class _FakePage:
    def goto(self, *a, **kw):
        raise Exception("Simulated navigation failure")
    def wait_for_selector(self, *a, **kw):
        pass
    def query_selector_all(self, *a, **kw):
        return []
    def inner_text(self, *a, **kw):
        return ""
    def title(self):
        return ""
    @property
    def url(self):
        return ""
    def is_closed(self):
        return False
    def click(self, *a, **kw):
        pass
    def fill(self, *a, **kw):
        pass
    def go_back(self, *a, **kw):
        pass

# Only inject fake page if browser is already open (avoids init errors)
if wb._page is not None:
    wb._page = _FakePage()
    brain2 = Brain()
    brain2.process_input("Search YouTube for Blender tutorials")
    wb._page = original_page  # restore
else:
    print("Browser not yet open — skipping failure simulation (init would also fail).")

print("\nAll tests complete.")
