from dotenv import load_dotenv
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

load_dotenv()

docs = [
            Document(page_content="Agentic AI systems can make auotomous decisions."),
            Document(page_content="RAG combines retrieval with LLM genertion."),
            Document(page_content="LangChain helps build LLM-powered applications.")
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs,embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOllama(model="gpt-4.1-mini" , temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Answer Using only the context:\n(context)\n\nQuestion:: (question)"
)

rag_chain = (
    {"context":retriever,"Question":RunnablePassthrough()}
    | prompt 
    | llm
)

print(rag_chain.invoke("what is RAG").content)

