from llm.langgraph_agent import run_voice_agent_turn


def main() -> None:
    queries = [
        "Bonjour",
        "Merci",
        "Quel est mon solde ?",
        "Qu'est-ce qu'une carte bancaire ?",
        "Explique la lettre de change",
    ]

    for q in queries:
        s = run_voice_agent_turn(q)
        print("Q:", q)
        print("tool_calls:", s.get("tool_calls"))
        print("tool_results:", s.get("tool_results"))
        print("rag_len:", len((s.get("rag_context") or "")))
        print(
            "agent_iterations:",
            s.get("agent_iterations"),
            "tool_call_count:",
            s.get("tool_call_count"),
        )
        print("---")


if __name__ == "__main__":
    main()
