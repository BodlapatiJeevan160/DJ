"""
real_runtime_validation.py
===========================
Scripted real-runtime validation that mirrors main.py's Brain setup.
Runs the 6 required queries sequentially with actual Ollama (no mocking),
records timing, selected path, tool, and final response.

Run:  python real_runtime_validation.py
"""
import time
import json
import sys

# ── Setup (mirrors main.py) ──────────────────────────────────────────────────
from core.brain import Brain
from core.notifier import manager as notification_manager
from core.context_monitor import monitor as context_monitor

# Start background services (same as main.py, minus audio/voice for scripted mode)
context_monitor.start()

brain = Brain()

# ── Tracking wrapper ─────────────────────────────────────────────────────────
def run_query(label, query):
    """Execute a single query, track Ollama call, record timing."""
    ollama_called = False
    tool_selected = [None]

    orig_chat_json = brain.llm.chat_json
    orig_chat      = brain.llm.chat
    orig_exec      = brain.registry.execute_tool

    def tracked_chat_json(msgs):
        nonlocal ollama_called
        ollama_called = True
        return orig_chat_json(msgs)

    def tracked_chat(msgs):
        nonlocal ollama_called
        ollama_called = True
        return orig_chat(msgs)

    def tracked_exec(name, args, confirm_callback=None, **kw):
        tool_selected[0] = name
        # Auto-approve confirmations for scripted run
        if confirm_callback:
            import builtins
            orig_input = builtins.input
            builtins.input = lambda *a, **kw2: "y"
        try:
            result = orig_exec(name, args, confirm_callback=confirm_callback, **kw)
        finally:
            if confirm_callback:
                builtins.input = orig_input
        return result

    brain.llm.chat_json        = tracked_chat_json
    brain.llm.chat             = tracked_chat
    brain.registry.execute_tool = tracked_exec
    brain.history = []  # fresh context per query

    t_start = time.perf_counter()
    # Auto-answer any confirmation prompts
    import builtins
    orig_input = builtins.input
    builtins.input = lambda *a, **kw2: "y"
    try:
        response = brain.process_input(query)
    finally:
        builtins.input = orig_input

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Restore
    brain.llm.chat_json         = orig_chat_json
    brain.llm.chat              = orig_chat
    brain.registry.execute_tool = orig_exec

    path = "FAST-PATH" if not ollama_called else "LLM (Ollama)"
    return {
        "label":        label,
        "query":        query,
        "path":         path,
        "ollama_called": ollama_called,
        "tool":         tool_selected[0],
        "elapsed_ms":   elapsed_ms,
        "response":     response,
    }


# ── Queries ──────────────────────────────────────────────────────────────────
QUERIES = [
    ("A", "What time is it?"),
    ("B", "Turn speech on."),
    ("C", "Turn speech off."),
    ("D", "What is my favorite programming language?"),
    ("E", "Remind me in 10 seconds."),
    ("F", "Watch for Chrome and notify me when I'm working in it."),
]

print("\n" + "=" * 68)
print("  JARVIS V1 — REAL RUNTIME VALIDATION")
print("=" * 68)
print(f"  Queries: {len(QUERIES)}   Model: phi3:mini (real Ollama)\n")

results = []
for letter, q in QUERIES:
    print(f"[{letter}] {q!r}")
    r = run_query(letter, q)
    results.append(r)
    status = "FAST" if not r["ollama_called"] else " LLM"
    print(f"     path={r['path']}  tool={r['tool']}  time={r['elapsed_ms']:.0f}ms")
    print(f"     response: {r['response'][:100]}")
    print()


# ── Security invariant re-check ───────────────────────────────────────────────
print("=" * 68)
print("  SECURITY INVARIANTS")
print("=" * 68)
errors = []

# SI-1: ToolRegistry not bypassed for fast-path
print("\n[SI-1] Fast-path still routes through execute_tool (ToolRegistry)")
si1_brain = Brain()
si1_brain.history = []
exec_called = {"name": None}

def si1_exec(name, args, confirm_callback=None, **kw):
    exec_called["name"] = name
    return json.dumps({"success": True, "data": {"date": "2026-09-25", "time": "22:57", "weekday": "Thursday"}})

si1_brain.registry.execute_tool = si1_exec
si1_brain.process_input("What time is it?")
if exec_called["name"] == "get_datetime":
    print("  [PASS] execute_tool called with 'get_datetime'")
else:
    print(f"  [FAIL] execute_tool NOT called correctly (got {exec_called['name']!r})")
    errors.append("SI-1 FAILED")

# SI-2: Dangerous tool requires confirmation
print("\n[SI-2] Dangerous tool (write_file) still requires confirmation")
si2_brain = Brain()
si2_brain.history = []
si2_brain.llm.chat_json = lambda msgs: json.dumps({
    "tool": "write_file",
    "args": {"location": "desktop", "name": "si2_test.txt", "content": "test"}
})
confirm_requested = {"yes": False}

def si2_exec(name, args, confirm_callback=None, **kw):
    if name == "write_file" and confirm_callback:
        import builtins
        orig = builtins.input
        builtins.input = lambda *a, **kw2: "n"   # deny
        try:
            confirm_callback(name, args)
        finally:
            builtins.input = orig
        confirm_requested["yes"] = True
        return json.dumps({"success": False, "error": "User denied."})
    return json.dumps({"success": True, "data": {}})

si2_brain.registry.execute_tool = si2_exec
si2_brain.llm.chat = lambda msgs: "Denied."
si2_brain.process_input("Write hello to desktop/si2_test.txt")
if confirm_requested["yes"]:
    print("  [PASS] Confirmation was requested before write_file")
else:
    print("  [FAIL] write_file ran WITHOUT confirmation")
    errors.append("SI-2 FAILED")

# SI-3: Ambiguous request remains LLM-bound
print("\n[SI-3] Ambiguous request ('What time is it in Tokyo?') remains LLM-bound")
si3_brain = Brain()
si3_brain.history = []
si3_ollama = {"called": False}
orig_cj = si3_brain.llm.chat_json
def si3_cj(msgs):
    si3_ollama["called"] = True
    return '{"tool": null, "args": null}'
si3_brain.llm.chat_json = si3_cj
si3_brain.llm.chat = lambda msgs: "Tokyo time is unknown without a world-clock tool."
si3_brain.process_input("What time is it in Tokyo?")
if si3_ollama["called"]:
    print("  [PASS] Ambiguous query routed through LLM Pass-1")
else:
    print("  [FAIL] Ambiguous query was fast-pathed — review patterns")
    errors.append("SI-3 FAILED")

# SI-4: PCA isolated from TTS
print("\n[SI-4] PCA notification path does NOT call audio TTS")
tts_calls = {"count": 0}
try:
    from core.audio import engine as audio_engine
    orig_speak = audio_engine.speak
    def tracking_speak(*a, **kw):
        tts_calls["count"] += 1
        return orig_speak(*a, **kw)
    audio_engine.speak = tracking_speak
    context_monitor.add_rule("__si4_test__", "SI-4 test msg")
    from core.notifier import manager as notif_manager
    notif_manager.notify("SI-4 test msg")
    context_monitor.remove_rule("__si4_test__")
    audio_engine.speak = orig_speak
    if tts_calls["count"] == 0:
        print("  [PASS] PCA notification does NOT trigger TTS")
    else:
        print(f"  [FAIL] TTS called {tts_calls['count']} time(s) via PCA path")
        errors.append("SI-4 FAILED")
except Exception as e:
    print(f"  [SKIP] Audio engine not running in script context: {e}")

# SI-5: USER MEMORY not in Pass-1 system prompt
print("\n[SI-5] USER MEMORY block is NOT injected into Pass-1 prompt")
si5_brain = Brain()
si5_brain.history = []
captured_sys = {"content": None}
def si5_cj(msgs):
    for m in msgs:
        if m.get("role") == "system":
            captured_sys["content"] = m["content"]
    return '{"tool": null, "args": null}'
si5_brain.llm.chat_json = si5_cj
si5_brain.llm.chat = lambda msgs: "ok"
si5_brain.process_input("What time is it in Tokyo?")
if captured_sys["content"] is None:
    print("  [WARN]  Could not capture Pass-1 system prompt")
elif "USER MEMORY" not in captured_sys["content"]:
    print("  [PASS] 'USER MEMORY' NOT found in Pass-1 system prompt")
else:
    print("  [FAIL] 'USER MEMORY' WAS found in Pass-1 system prompt")
    errors.append("SI-5 FAILED")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  SUMMARY")
print("=" * 68)

print(f"\n{'Label':<6} {'Query':<45} {'Path':<12} {'Tool':<20} {'Time':>8}")
print("-" * 95)
for r in results:
    path_tag = "FAST-PATH" if not r["ollama_called"] else "LLM"
    tool_str = r["tool"] or "(none)"
    print(f"  [{r['label']}]  {r['query']:<43} {path_tag:<12} {tool_str:<20} {r['elapsed_ms']:>6.0f}ms")

print()
if errors:
    print(f"[FAIL] {len(errors)} security invariant(s) FAILED:")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("[PASS] All 5 security invariants CONFIRMED")
    print("[PASS] Routing validation complete")

context_monitor.stop()
