import llm.langgraph_agent as lga

orig = lga.generate_ai_response_with_system_schema


def wrapper(system_text: str, user_text: str, schema: dict) -> str:
    print("--- SYSTEM PROMPT (first 300) ---")
    print((system_text or "")[:300])
    print("system_len=", len(system_text or ""))
    print("--- USER PROMPT (first 300) ---")
    print((user_text or "")[:300])
    print("user_len=", len(user_text or ""))
    print("--- SCHEMA (keys) ---")
    try:
        print(list((schema or {}).keys()))
    except Exception:
        print("<unprintable schema>")
    out = orig(system_text, user_text, schema)
    print("--- LLM OUTPUT (first 400) ---")
    print((out or "")[:400])
    print("--- LLM OUTPUT (last 200) ---")
    print((out or "")[-200:])
    print("len=", len(out or ""))
    return out


lga.generate_ai_response_with_system_schema = wrapper

r = lga.run_voice_agent_turn("quel est mon solde?")
print("FINAL tool_calls=", r.get("tool_calls"))
print("FINAL tool_results=", r.get("tool_results"))
print("FINAL response_text=", r.get("response_text"))
