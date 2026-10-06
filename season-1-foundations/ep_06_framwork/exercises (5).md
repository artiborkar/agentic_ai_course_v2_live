# Ep 06 — Exercises

1. Add a second node `summarize` that shortens the answer; wire `answer -> summarize -> END`.
2. Add a `style: str` field to `State` and use it in the prompt (e.g. "explain like I'm 10").
3. Draw your Ep 05 raw agent as a graph (state, nodes, edges) on paper.
4. Write a one-paragraph decision guide: which of langchain/langgraph/deepagents you'd use for (a) a quick prototype, (b) a production support agent, (c) a multi-hour autonomous research task.
5. **Interview:** "Why LangGraph over a plain chain?" — answer mentioning control flow, state, persistence, branching.

## Solution hints
- 4: (a) langchain, (b) langgraph, (c) deepagents.
- 5: chains are linear; LangGraph gives branching/loops, first-class state, checkpointing/persistence, HITL — production control.
