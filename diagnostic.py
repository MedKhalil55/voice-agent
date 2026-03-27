#!/usr/bin/env python
"""Diagnostic: check LLM raw output for a card query."""

from llm.agent import generate_ai_response

query = "donne moi Exemples de cartes bancaires commercialisées sur les banques locales: Par Attijari Bank"
rag_ctx = """VI-Exemples de cartes bancaires commercialisées sur les banques locales: Par Attijari Bank : 
1- Cartes GOLD Nationale et Internationale : Vous êtes client privilégié, haut cadre, diplomate, dirigeant ou chef d'entreprise, commerçant ou professionnel libéral; la banque vous offre entre autres:

Des paiements sur internet auprès des sites affiliés à: www.clicktopay.com.tn

Des privilèges auprès de différentes enseignes (hôtels, agences de voyages, compagnies de location de voitures, restaurants etc.).
"""

system_prompt = """You are a French voice AI assistant for banking queries.
Your role is to help users with banking information and operations.

DECISION FRAMEWORK:
1. Use RAG context (banking documents) as the primary source of truth when available
2. Decide if a tool call is needed based on the user's intent and available context
3. Always respond in complete, natural French sentences
4. Be concise but thorough

OUTPUT FORMAT (MUST BE VALID JSON):
{
  "action": "tool" or "respond",
  "tool_name": "tool_name" (required only if action="tool", else null),
  "arguments": {...} (required only if action="tool", else {}),
  "response": "your answer" (required only if action="respond", else "")
}

IMPORTANT RULES:
- If action="tool": must specify tool_name and arguments; response must be empty string
- If action="respond": must provide complete response; tool_name must be null
- Always output ONLY valid JSON, nothing else
- No explanations, no markdown, pure JSON
"""

user_prompt = f"""User question:
{query}

Banking documents (RAG context):
{rag_ctx}

Previous tool executions:
[No tool results yet]

Output JSON response:"""

full_prompt = f"{system_prompt}\n{user_prompt}"

print("=" * 70)
print("SENDING TO LLM")
print("=" * 70)
print(f"Prompt length: {len(full_prompt)} chars")
print()

output = generate_ai_response(full_prompt)

print("=" * 70)
print("RAW LLM OUTPUT")
print("=" * 70)
print(f"Length: {len(output)} chars")
print()
print(output)
print()

if len(output) > 200:
    print("=" * 70)
    print("LAST 200 CHARS:")
    print("=" * 70)
    print(output[-200:])
