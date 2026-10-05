from concurrent.futures import Executor
import os 
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage , ToolMessage , SystemMessage
from langchain.chat_models import init_chat_model

load_dotenv()
MODEL = os.getenv("GROQ_MODEL","groq:qwen/qwen3.8-27b")

MAX_STEPS = 5


@tool 
def search(query :str) -> str:
    """Look up a fact (mocked for the demo)."""
    facts = {
                "population of japan" : "Japan's population is about 124 million.",
                "population of india" : "India's population is about 1.43 billion.",

            }
    for k , v in facts.items():
        if k in query.lower():
            return v
    return "No result found (mocked search)."


@tool
def calculator(expression:str) -> str:
    """Evaluate basic artithmetric , e.g. '1240000000/2"""
    allowed = set("0123456789+-*/().e")
    if not set(expression) <= allowed:
        return "Error: only basic arithmetic allowed."

    try:
        return str(eval(expression))    #safe: restricted charset ##eval(2+5*2)
    except Exception:
        return "Error: invaled expression."

TOOLS = {t.name: t for t in [search,calculator]}

def run_tool(name: str ,args: dict) ->str:
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error unknown tool'{name}'."
    try:
        return fn.invoke(args)
    except Exception as exc:
        return f"Error runing {name}: {type(exc).__name__}."

def run_agent(question:str) ->str:
    llm = init_chat_model(MODEL , temperature=0.1).bind_tools(list(TOOLS.values()))

    messages = [
                    SystemMessage(content="You Are a Helpful Agent.Use tools When Needed,then give a final answer."),
                    HumanMessage(content=question),

                ]

    print(f"[bold blue]User:[/bold blue] {question}\n")

    for step in range(1, MAX_STEPS):
        ai = llm.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            print(f"[dim](finished in {step} step(5)[/dim])")
            return ai.content

        for tc in ai.tool_calls:
            print(f"[yellow]Step {step} - REASON_ACT:[/yellow] {tc['name']}({tc['args']})")
            result = run_tool(tc['name'],tc['args'])
            print(f"[green]  OBSERVE:[/green] {result}")
            messages.append(ToolMessage(content=result,tool_call_id=tc['id']))

    return "Stopped: reached the max step limit (bounded autonomy safeguard)."


def main():
    question = "What is the population of japan divied by 2?"
    
    responce = run_agent(question)
    print(f"\n[bold green]Agent:[/bold green]{responce}")
    

    question2= "What is the population of India"
    responce2= run_agent(question2)
    print(f"\n[bold green]Agent:[/bold green] {responce2}")

if __name__ == "__main__":
    main()

