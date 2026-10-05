# (Optional, free) Set DEMO_MODEL=ollama:llama3.2 and re-run.
#  Note any difference in how it uses tools.


"""
Ep 01 DEMO — watch a real AI AGENT (not a chatbot) work.

We are NOT explaining how to build this yet (that's Ep 02-07). For now, just
RUN it and watch the loop: the agent decides on its own whether to calculate
or search, runs the tool, reads the result, and answers.

Official docs (verify before teaching — June 2026):
  - create_agent:  https://docs.langchain.com/oss/python/langchain/agents
  - tools (@tool): https://docs.langchain.com/oss/python/langchain/tools

Run:
    python demo_agent.py

Model: defaults to OpenAI. For a FREE local run, install Ollama
(`ollama pull llama3.2`) and set DEMO_MODEL=ollama:llama3.2 in your .env.

The code is split into numbered STEPS that match the README walkthrough.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from rich import print
import time
from langchain_core.tools import tool
from langchain.agents import create_agent   # v1 API (replaces create_react_agent)

# STEP 1 — Load secrets from .env (never hardcode/log API keys).
load_dotenv()
MODEL = os.getenv("DEMO_MODEL", "ollama:llama3.2")


# STEP 2 — Give the agent its first "hand": a calculator tool.
# The function name + docstring + type hints become the tool's description,
# which the model reads to decide WHEN to use it.
@tool
def calculator(expression: str) -> str:
    """Do basic math. Example: '23 * 19 + 7'."""
    allowed = set("0123456789+-*/(). ")
    # Safety: only allow math characters; never eval untrusted free text.
    if not set(expression) <= allowed:
        return "Error: only basic math is allowed."
    return str(eval(expression))  # safe because input is restricted above


# STEP 3 — Give it a second hand: a (mock) search tool, so the demo needs no key.
@tool
def search(query: str) -> str:
    """Look up a simple fact on the web."""
    facts = {"capital of australia": "Canberra is the capital of Australia."}
    for key, value in facts.items():
        if key in query.lower():
            return value
    return "No result found (this search is mocked for the demo)."


def main() -> None:
    # STEP 4 — Create the agent: a model + its tools. That's it.
    # We tell it to ALWAYS use the tools so the loop is visible in the demo.
    agent = create_agent(
        model=MODEL,
        tools=[calculator, search],
        system_prompt=(
            "You are a tool-using agent. You MUST use the calculator tool for any "
            "arithmetic and the search tool for any factual lookup. Never answer "
            "math or facts from memory — always call the right tool first, then answer."
        ),
    )

    # STEP 5 — Ask one question that needs BOTH tools.
    question = "What is the capital of Australia, and what is 23 * 19 + 7?"
    print(f"[bold cyan]USER:[/bold cyan] {question}\n")
    print("[dim]--- watch the loop: the agent reasons, acts (tool), observes, repeats ---[/dim]\n")

    # STEP 6 — Stream each step so students SEE the loop (not just the final answer).
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        last = chunk["messages"][-1]
        kind = last.__class__.__name__
        if kind == "HumanMessage":
            continue  # this is just our question echoed back — skip it
        if getattr(last, "tool_calls", None):
            for call in last.tool_calls:
                print(f"[yellow]AGENT decides -> call tool:[/yellow] {call['name']}({call['args']})")
                time.sleep(3)
        elif kind == "ToolMessage":
            print(f"[green]TOOL returns:[/green] {last.content}")
        elif last.content:
            print(f"[bold magenta]AGENT answers:[/bold magenta] {last.content}")
            time.sleep(3)


if __name__ == "__main__":
    main()