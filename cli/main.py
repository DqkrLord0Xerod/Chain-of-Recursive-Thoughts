from __future__ import annotations

from core.chat_v2 import CoRTConfig
from core.recursive_engine_v2 import create_optimized_engine
from config import settings
from core.security import CredentialManager


async def main() -> None:
    """Run the recursive thinking chat CLI.

    Commands:
        exit                 Quit the program.
        save                 Save the conversation history to ``conversation.json``.
        history [session_id] Show saved loop history for a session.
    """
    print("🤖 Recursive Thinking Chat v2")
    print("=" * 50)

    manager = CredentialManager()
    api_key = settings.openrouter_api_key or manager.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "Error: OPENROUTER_API_KEY not set. Provide it via env or secrets file"
        )
        return

    config = CoRTConfig(api_key=api_key, model=settings.model)
    engine = create_optimized_engine(config)

    session_id = input("Session ID (default 'default'): ").strip() or "default"

    print("\nChat initialized! Type 'exit' to quit, 'save' to save conversation.")
    print("The AI will think recursively before each response.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "save":
            await engine.save_conversation("conversation.json")
            continue
        if user_input.lower().startswith("history"):
            parts = user_input.split()
            sid = parts[1] if len(parts) > 1 else session_id
            history = await engine.loop_controller.load_loop_history(sid)
            if not history:
                print("No history available.")
            else:
                for i, state in enumerate(history, 1):
                    print(
                        f"Run {i}: rounds={len(state.rounds)}, "
                        f"reason={state.convergence_reason}"
                    )
            continue
        if not user_input:
            continue

        result = await engine.think_and_respond(
            user_input, session_id=session_id
        )
        print(f"\n🤖 AI FINAL RESPONSE: {result.response}\n")
        print("\n--- COMPLETE THINKING PROCESS ---")
        for item in result.thinking_history:
            label = "[SELECTED]" if item.selected else "[ALTERNATIVE]"
            print(f"\nRound {item.round_number} {label}:")
            print(f"  Response: {item.response}")
            if item.explanation and item.selected:
                print(f"  Reason for selection: {item.explanation}")
            print("-" * 50)
        print("--------------------------------\n")

    save_on_exit = input("Save conversation before exiting? (y/n): ").strip().lower()
    if save_on_exit == "y":
        await engine.save_conversation("conversation.json")
    print("Goodbye! 👋")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
