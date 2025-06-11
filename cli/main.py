from __future__ import annotations

from core.chat_v2 import CoRTConfig, create_default_engine
from config import settings


async def main() -> None:
    """Run the recursive thinking chat CLI.

    Commands:
        exit      Quit the program.
        save      Save the conversation history to ``conversation.json``.
    """
    print("🤖 Recursive Thinking Chat v2")
    print("=" * 50)

    api_key = settings.openrouter_api_key
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set. Please export it or add to .env")
        return

    config = CoRTConfig(api_key=api_key, model=settings.model)
    engine = create_default_engine(config)

    print("\nChat initialized! Type 'exit' to quit, 'save' to save conversation.")
    print("The AI will think recursively before each response.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "save":
            await engine.save_conversation("conversation.json")
            continue
        if not user_input:
            continue

        result = await engine.think_and_respond(user_input)
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
