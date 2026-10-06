import os 
from pathlib import Path
import random
from dotenv import load_dotenv
from rich import print
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage , ToolMessage , SystemMessage
from langchain.chat_models import init_chat_model
load_dotenv()
MODEL = os.getenv("GROQ_MODEL","groq:qwen/qwen3.8-27b")

MAX_STEPS = 4
# MAX_STEPS = 1


@tool 
def search(query :str) -> str:
    """Look up a fact (mocked for the demo)."""

    if random.random() < 0.5:
        raise RuntimeError("Search tool failed ramdomly.")

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



@tool
def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency using mocked exchange rates."""

    rates = {
        ("USD", "INR"): 88.0,
        ("INR", "USD"): 1 / 88.0,
        ("EUR", "INR"): 103.0,
        ("INR", "EUR"): 1 / 103.0,
    }

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    key = (from_currency, to_currency)

    if key not in rates:
        return f"Error: conversion from {from_currency} to {to_currency} is not supported."

    converted = amount * rates[key]

    return f"{converted:.2f} {to_currency}"

TOOLS = {t.name: t for t in [search,calculator,currency_convert]}

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
        print(f"[cyan]Step {step} - Message count: {len(messages)}[/cyan]")

        ai = llm.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            print(f"[dim](finished in {step} step(5)[/dim])")
            return ai.content

        for tc in ai.tool_calls:
            print(f"[yellow]Step {step} - REASON+ACT:[/yellow] {tc['name']}({tc['args']})")
            result = run_tool(tc['name'],tc['args'])
            print(f"[green]  OBSERVE:[/green] {result}")
            messages.append(ToolMessage(content=result,tool_call_id=tc['id']))

    return "Stopped: reached the max step limit (bounded autonomy safeguard)."


def main():
    # question = "What is the population of japan divied by 2?"
    
    # responce = run_agent(question)
    # print(f"\n[bold green]Agent:[/bold green]{responce}")
    

    # question2= "What is the population of India"
    # responce2= run_agent(question2)
    # print(f"\n[bold green]Agent:[/bold green] {responce2}")

    # question3= "Search for the population of India, convert 100 USD to INR, and calculate the result multiplied by 2."
    # responce3= run_agent(question3)
    # print(f"\n[bold green]Agent:[/bold green] {responce3}")

    question4= "Search for the population of India, convert 100 USD to INR, and calculate the result × 2."
    responce4= run_agent(question4)
    print(f"\n[bold green]Agent:[/bold green] {responce4}")




    # while True:
    #     print("[bold cyan]USER > [/bold cyan]",end="")
    #     user_input = input()
    #     if user_input.lower() == "exit":
    #         break
    #     responce = run_agent(user_input)
    #     print(f"\n[bold green] Agent > [/bold green] {responce}")


if __name__ == "__main__":
    main()

