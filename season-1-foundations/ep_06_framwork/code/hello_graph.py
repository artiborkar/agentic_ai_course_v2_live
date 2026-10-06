import os
from dotenv import load_dotenv
from rich import print
from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph ,START, END

load_dotenv()
MODEL = os.getenv("GROQ_MODEL","groq:qwen/qwen3.8-27b")


class State(TypedDict):
    question : str
    answer : str

llm = init_chat_model(MODEL)

def answer_node(state:State) -> dict:
    response = llm.invoke(state["question"])
    return {"answer":response.content}

def build_graph():
    graph = StateGraph(State)

    graph.add_node("answer",answer_node)
    graph.add_edge(START,"answer")
    graph.add_edge("answer",END)

    return graph.compile()

def main():
    agent = build_graph()

    agent_result = agent.invoke({"question":"what is the python , explain in single sentence?"})

    print(f"[bold cyan]Que1:[/bold cyan] {agent_result['question']}")
    print(f"[bold green]Ans1:[/bold green] {agent_result['answer']}")
    

    agent_result_2 = agent.invoke({"question": "what is the agentic ai , explain in single sentence?" })

    print(f"[bold cyan]Que2:[/bold cyan] {agent_result_2['question']}")
    print(f"[bold green]Ans2:[/bold green] {agent_result_2['answer']}")


if __name__ == "__main__":
    main()