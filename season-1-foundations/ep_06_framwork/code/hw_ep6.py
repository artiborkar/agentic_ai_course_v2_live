import os
from rich import print
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import TypedDict
from langgraph.graph import StateGraph ,START, END

load_dotenv()

MODEL = os.getenv("GROQ_MODEL","groq:qwen/qwen3.8-27b")

class State(TypedDict):
    question : str
    answer : str
    summarize : str
    style : str

llm = init_chat_model(MODEL,temperature=0)

def answer_node(state : State) -> dict:
    responce = llm.invoke(f"{state['style']}\n Question : {state['question']}")  
    return {"answer" : responce.content}

def summarize_node(state : State) -> dict :
    responce = llm.invoke(f"Summarize this answer in one short sentence:\n{state['answer']}")
    return {"summarize":responce.content}


def build_graph():
    graph = StateGraph(State)

    graph.add_node("answer" , answer_node)
    graph.add_node("summarize" , summarize_node)

    graph.add_edge(START , "answer")
    graph.add_edge("answer" , "summarize")
    graph.add_edge("summarize" , END)

    return graph.compile()


def main():
    agent = build_graph()
    agent_result = agent.invoke({"question":"what is the python ?","style": "Explain like I'm 10"})
    print(agent_result)

if __name__ == "__main__":
    main()