# Ep 04 — Exercises

1. Add a `currency_convert(amount: float, from_ccy: str, to_ccy: str)` tool (mock rates). Ask a question that needs it.
2. Ask `lookup_user` or `get_resume` for an id or candidate that doesn't exist (e.g. `'u99'` or `'Rahul'`). Confirm the agent gets a clean "not found" message back, not a crash.
3. Make a tool that raises an exception internally; confirm `run_tool_safely` returns a clean message and the agent still answers.
4. Make one tool's docstring vague. Ask an ambiguous question. Observe a wrong/no tool pick. Fix the docstring.
5. Use a **Pydantic model** as a tool's argument schema (connect to Ep 03).
6. **Interview:** "Walk me through what happens when an LLM uses a tool." — answer in 4 steps.

## Solution hints
- 6: (1) tools described to model, (2) model emits a tool call with args, (3) your code validates+runs it, (4) result fed back; model answers or calls again.
- Good descriptions + input validation + try/except + returning errors-as-text are the production essentials.
