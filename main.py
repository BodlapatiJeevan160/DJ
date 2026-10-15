import logging
import sys
from core.brain import Brain
from core.audio import engine as audio_engine
from core.voice_input import engine as voice_engine
from core.notifier import manager as notification_manager
from core.context_monitor import monitor as context_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("jarvis.log"),
        # We don't necessarily want all INFO logs spamming the interactive terminal, 
        # so we keep file handler at INFO and stream handler at WARNING, but for now 
        # let's set stream to ERROR to keep terminal clean for the chat interface.
        logging.StreamHandler(sys.stdout)
    ]
)

# Quiet down HTTP/Stream logs to avoid clutter
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("core.brain").handlers = []
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)
logging.getLogger().addHandler(console_handler)
# Remove the root default stream handler to prevent double logging in console
logging.getLogger().handlers = [h for h in logging.getLogger().handlers if not isinstance(h, logging.StreamHandler)]
logging.getLogger().addHandler(console_handler)

def main():
    print("Initializing JARVIS Local AI...")
    try:
        brain = Brain()
        audio_engine.start()
        voice_engine.start(brain, audio_engine)
        voice_engine.toggle(True)  # enable continuous listening by default
        context_monitor.start()
    except Exception as e:
        print(f"Failed to initialize Brain: {e}")
        return

    print("JARVIS is online. Using local model 'phi3:mini'.")
    print("Type 'exit' or 'quit' to shut down.")

    # TTS startup diagnostic — surfaces engine failures immediately.
    if not audio_engine.running:
        print(f"[JARVIS WARNING] TTS worker failed to start — voice responses will be silent.")
        if audio_engine.init_error:
            print(f"  Reason: {audio_engine.init_error}")
    elif not audio_engine.enabled:
        print("[JARVIS WARNING] TTS is disabled — voice responses will be silent.")
    else:
        print("[JARVIS] TTS ready — voice responses will be spoken.")
        audio_engine.speak("JARVIS.")

    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("JARVIS: Shutting down. Goodbye!")
                audio_engine.enabled = False
                break
            
            if not user_input.strip():
                continue
                
            response = brain.process_input(user_input)
            print(f"\nJARVIS: {response}")
            audio_engine.speak(response)
            
        except KeyboardInterrupt:
            print("\nJARVIS: Shutting down via interrupt. Goodbye!")
            audio_engine.stop_speaking()
            break
        except Exception as e:
            print(f"\nJARVIS: An unexpected error occurred: {e}")

    voice_engine.stop()
    audio_engine.stop()
    context_monitor.stop()

if __name__ == "__main__":
    main()
