import os
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from langchain_core.tools import tool   ## hat pair hame tool deta hai
from langchain_core.messages import HumanMessage , ToolMessage  ## tool ka massage show karne ke liye hamne ToolMessage use kiya hai
from langchain.chat_models import init_chat_model           ## large model ko call karne ke liye use kiya hai

load_dotenv()
MODEL = os.getenv("GROQ_MODELS" , "groq:qwen/qwen3.8-27b")

RESUME_DIR = (Path(__file__).parent.parent / "resumes")   ##__file__->path ko pta chalta hai ki konsi directry hai .py file mai.

@tool
def add(a: float , b:float) -> float :
    """ Add function use for the addtion of two number """
    return a + b 

@tool
def get_weather(city : str) ->str:
    """ Get the current weather of given city"""
    weather = {"mumbai" : "32c , humid" , "delhi" : "38c , haty" , "bangaluru" : "26c ,plesent"}
    return f"City {city} - {weather.get(city.lower() , f"No Weather data for {city}")}"

@tool
def lookup_user(user_id:str)->str:
    """Look up a user and plan by their id (i.e,'u1')."""
    users = {"u1" : "Arti (pro plan)" , "u2" : "Diya (free plan)"}
    return users.get(user_id , f"No user with id '{user_id}'.")

@tool
def get_resume(name:str)->str:
    """ Read and return store resume for candidate name (ramesh ,khushi) """
    file_name = name.lower().strip().split()[0]
    file_path = RESUME_DIR / f"{name}.txt"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return f"No resume found for {name}"

TOOLS = {t.name: t for t in [add , get_weather , lookup_user , get_resume]}

def run_tool_safely(name: str ,args :dict) -> str:
    tool_fn = TOOLS.get(name ,None)

    if tool_fn is None:
        return f"Error unknown tool named {name}."

    try:
        return tool_fn.invoke(args)
    except Exception as e:
        return f"Error runing tools {name} : {e}"


def main() -> None:
    llm = init_chat_model(MODEL , temperature=0)
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

    question = 'can we select priya for Python developer interview'
    messages =[HumanMessage(content=question)]
    print(f"[bold cyan] USER : [/bold cyan] {question} \n")

    ai = llm_with_tools.invoke(messages)
    messages.append(ai)

    if not ai.tool_calls:
        print(f"[magenta] AI:[/magenta] {ai.content}")
        return None

    for tc in ai.tool_calls:
        print(f"[yellow] Tool call:[/yellow] {tc['name']}({tc['args']})")
        result = run_tool_safely(tc["name"],tc["args"])
        print(f"[green]Result:[/green] {result}")
        messages.append(ToolMessage(content=result,tool_call_id=tc["id"]))


    final = llm_with_tools.invoke(messages)
    print(f"\n[bold magenta]AI :[/bold magenta] {final.content}")

if __name__ == "__main__":
    main()