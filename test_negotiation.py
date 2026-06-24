"""Test the voice agent logic via keyboard input — no microphone needed."""

from main import VoiceAgent
from unittest.mock import MagicMock

# Patch STT so no microphone is needed
agent = VoiceAgent()
agent._stt = MagicMock()
agent._stt.pause = MagicMock()
agent._stt.resume = MagicMock()
agent._stt.start_stream = MagicMock()
agent._stt.stop_stream = MagicMock()

# Patch speak so it prints instead of playing audio
def fake_speak(text, log_output=True):
    print(f"\n🤖 AGENT: {text}\n")

agent.speak = fake_speak

# Start the agent (greeting will print, no audio)
agent.start()

print("\n" + "="*50)
print("TEXT MODE — type your messages, press Enter")
print("Type 'quit' to exit")
print("="*50 + "\n")

# Simulate conversation turns
while not agent._shutdown_event.is_set():
    try:
        user_input = input("👤 YOU: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            agent.shutdown()
            break
        # Inject text directly as if STT recognized it
        agent.handle_final(user_input)
        # Wait for response to finish
        import time
        time.sleep(0.5)
    except KeyboardInterrupt:
        agent.shutdown()
        break