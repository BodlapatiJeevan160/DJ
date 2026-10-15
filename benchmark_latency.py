"""
Latency benchmark — measures fast-path vs LLM-bound requests.
All registry.execute_tool calls are mocked so Ollama is only hit when Pass-1 is needed.
"""
import time
import json
from unittest.mock import MagicMock, patch
from core.brain import Brain

# Shared mock tool result that looks like a valid get_datetime response
_DATETIME_RESULT = json.dumps({"success": True, "data": {"date": "2026-09-18", "time": "22:36", "weekday": "Friday"}})
_GENERIC_RESULT  = json.dumps({"success": True, "data": {}})

def make_brain():
    b = Brain()
    # Mock tool execution so we never do real side effects
    def _exec(name, args, **kw):
        if name == "get_datetime":
            return _DATETIME_RESULT
        if name == "toggle_speech":
            return json.dumps({"success": True, "data": {"enabled": args.get("enabled")}})
        return _GENERIC_RESULT
    b.registry.execute_tool = _exec
    return b

QUERIES = [
    ("A. What time is it?",               "What time is it?"),
    ("B. Fav programming language?",      "What is my favorite programming language?"),
    ("C. Turn speech on.",                "Turn speech on."),
    ("D. Remind me in 10 seconds.",       "Remind me in 10 seconds."),
    ("E. Why is the sky blue?",           "Why is the sky blue?"),
]

print("=" * 60)
print("AFTER benchmark (with fast-path + no memory in Pass-1)")
print("=" * 60)

for label, query in QUERIES:
    brain = make_brain()
    # For LLM-bound queries, mock LLM to return instantly
    brain.llm.chat_json = MagicMock(return_value='{"tool": null, "args": null}')
    brain.llm.chat     = MagicMock(return_value="Mocked LLM answer.")

    t0 = time.perf_counter()
    brain.process_input(query)
    elapsed = time.perf_counter() - t0
    print(f"{label:40s}  total={elapsed*1000:.1f}ms")

print("=" * 60)
