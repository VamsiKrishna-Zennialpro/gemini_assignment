"""
retrieval_agent_gemini_fixed.py
Compatible with LangChain v0.2+
Self-healing hybrid: Gemini → HuggingFace fallback for embeddings
"""

import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError


# ✅ Load environment and set correct API variable
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# --------------------------
# 1) Load documents
# --------------------------
def load_documents_from_dir(path: str) -> List[Document]:
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(path, filename), encoding="utf8")
            docs.extend(loader.load())
    return docs


# --------------------------
# 2) Chunk / preprocess
# --------------------------
def chunk_documents(docs: List[Document], chunk_size=800, chunk_overlap=100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for d in docs:
        for t in splitter.split_text(d.page_content):
            all_chunks.append(Document(page_content=t, metadata=d.metadata))
    return all_chunks


# --------------------------
# 3) Build Vectorstore with fallback
# --------------------------
def build_vectorstore(docs: List[Document], persist_dir="chroma_db"):
    try:
        print("🔹 Using Gemini embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
        vectordb.persist()
        print("✅ Gemini embeddings worked.")
        return vectordb

    except Exception as e:
        print("⚠️ Gemini embeddings failed, switching to HuggingFace.")
        print("Error:", e)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
        vectordb.persist()
        print("✅ HuggingFace embeddings ready.")
        return vectordb


# --------------------------
# 4) Wrap retriever as a Tool
# --------------------------
def make_retriever_tool(vectordb) -> Tool:
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return create_retriever_tool(
        retriever,
        name="document_retriever",
        description="Retrieve relevant document excerpts from the knowledge base."
    )


# --------------------------
# 5) Create Agent with Gemini + Tools
# --------------------------
def build_agent_with_tools(retriever_tool: Tool):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    @tool("calculator", "Perform simple arithmetic operations. Input: a math expression string")
    def calculator(expr: str) -> str:
        try:
            result = eval(expr, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    tools = [retriever_tool, calculator]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3
    )


# --------------------------
# 6) Run example queries
# --------------------------
def main():
    doc_dir = "./docs"
    os.makedirs(doc_dir, exist_ok=True)

    # Add sample docs if not present
    if not os.listdir(doc_dir):
        with open(os.path.join(doc_dir, "python_info.txt"), "w", encoding="utf8") as f:
            f.write("Python is a high-level programming language created by Guido van Rossum.")
        with open(os.path.join(doc_dir, "langchain_notes.txt"), "w", encoding="utf8") as f:
            f.write("LangChain is a framework for building LLM applications using tools and agents.")

    print("Loading and preparing documents...")
    docs = load_documents_from_dir(doc_dir)
    chunks = chunk_documents(docs)

    print("Building Chroma vectorstore...")
    vectordb = build_vectorstore(chunks)
    retriever_tool = make_retriever_tool(vectordb)

    print("Creating Gemini-powered retrieval agent...")
    agent = build_agent_with_tools(retriever_tool)

    # Demo queries
    queries = [
        "Who created Python?",
        "What is LangChain used for?",
        "What is 25 * 14?"
    ]

    for q in queries:
        print(f"\n=== QUERY: {q} ===")
        print(agent.run(q))
        print("===================\n")


if __name__ == "__main__":
    main()
