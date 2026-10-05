from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama (
                    model = "llama3.2",
                    temperature=0

                )

responce = llm.invoke([

                    ("system", " You are a helpful assitant."),
                    ("user", "Expalin Agentic AI in one sentence.")

                    ])

print(responce.text)