# Ep 05 — Exercises

1. Add a `currency_convert` tool and ask a 3-step question (search → convert → calculate).
2. Set `MAX_STEPS = 1` and ask a multi-step question. Confirm it stops safely at the cap.
3. Add a `print` of the running message count each step — watch "memory" grow.
4. Make `search` randomly fail ~50% of the time; confirm the agent recovers via the error message.
5. Write down 3 problems you personally hit — these are the exact problems LangGraph solves next.
6. **Interview:** "Implement a basic agent loop." — be able to write the reason→act→observe→repeat loop with a stop condition from memory.

## Solution hints
- The stop condition is "no tool calls in the AI message" OR "max steps reached."
- Memory = the ordered message list (System, Human, AI, Tool, AI, ...).
