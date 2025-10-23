# main_memory_rag.py
import os
import datetime
import time
from typing import List

# LangChain / loaders / vectorstore / embeddings / llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader, PDFMinerLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document

# DB (SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, Float, Text, TIMESTAMP
from sqlalchemy.orm import declarative_base, sessionmaker

# === CONFIGURATION ===
DATA_DIR = "data"
PERSIST_DIR = "vectorstore_ollama"
OLLAMA_MODEL = "llama3"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 6

# PostgreSQL connection
DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"

# Memory / summarization params
RECENT_HISTORY_LIMIT = 5
SUMMARY_INTERVAL = 10

# === PATCH NUMPY FOR COMPATIBILITY (NumPy 2.0) ===
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
    np.int_ = np.int64
    np.uint = np.uint64

# === DATABASE SETUP ===
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    user_prompt = Column(Text)
    assistant_response = Column(Text)
    temperature = Column(Float)
    top_p = Column(Float)
    max_tokens = Column(Integer)
    timestamp = Column(TIMESTAMP, default=datetime.datetime.utcnow)

class MemorySummary(Base):
    __tablename__ = "memory_summary"
    id = Column(Integer, primary_key=True)
    summary_text = Column(Text)
    timestamp = Column(TIMESTAMP, default=datetime.datetime.utcnow)

# --- PostgreSQL setup ---
Session = None
try:
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    print("✅ Connected to PostgreSQL.")
except Exception as e:
    print(f"❌ PostgreSQL connection failed: {e}\nProceeding without DB logging/memory (Session=None).")

# === DATABASE HELPERS ===
def log_interaction(prompt: str, response: str, temperature: float, top_p: float, max_tokens: int):
    """Store chat interaction in PostgreSQL."""
    if not Session:
        print("⚠️ Skipping DB logging (PostgreSQL not connected).")
        return
    session = Session()
    try:
        entry = ChatHistory(
            user_prompt=prompt,
            assistant_response=response,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        session.add(entry)
        session.commit()
    except Exception as e:
        print(f"⚠️ Failed to log to PostgreSQL: {e}")
        session.rollback()
    finally:
        session.close()

def load_recent_history(limit: int = RECENT_HISTORY_LIMIT) -> List[ChatHistory]:
    if not Session:
        return []
    session = Session()
    try:
        rows = session.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).limit(limit).all()
        return list(reversed(rows))
    except Exception as e:
        print(f"⚠️ Could not load recent history: {e}")
        return []
    finally:
        session.close()

def show_recent_chats(limit: int = RECENT_HISTORY_LIMIT):
    history = load_recent_history(limit)
    if not history:
        print("🕒 No recent chat history found.")
        return
    print(f"\n🕒 Recent {len(history)} interactions:")
    for h in history:
        ts = h.timestamp.strftime("%Y-%m-%d %H:%M:%S") if h.timestamp else ""
        print(f"[{ts}] You: {h.user_prompt}\n    Assistant: {h.assistant_response}\n")

def get_memory_summary() -> str:
    if not Session:
        return ""
    session = Session()
    try:
        last = session.query(MemorySummary).order_by(MemorySummary.timestamp.desc()).first()
        return last.summary_text if last else ""
    except Exception as e:
        print(f"⚠️ Failed to load memory summary: {e}")
        return ""
    finally:
        session.close()

def save_memory_summary(summary_text: str):
    if not Session:
        return
    session = Session()
    try:
        entry = MemorySummary(summary_text=summary_text)
        session.add(entry)
        session.commit()
        print("💾 Updated summarized memory in PostgreSQL.")
    except Exception as e:
        print(f"⚠️ Failed to save memory summary: {e}")
        session.rollback()
    finally:
        session.close()

# === DOCUMENT PROCESSING ===
def load_documents(data_dir=DATA_DIR):
    """Load PDFs and text files with reliable text extraction."""
    docs = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        file_docs = []

        if file.lower().endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(path)
                file_docs = loader.load()
                print(f"✅ Loaded text from {file} using PyMuPDF ({len(file_docs)} pages)")
            except Exception as e:
                print(f"⚠️ PyMuPDF failed on {file}, trying PDFMiner: {e}")
                try:
                    loader = PDFMinerLoader(path)
                    file_docs = loader.load()
                    print(f"✅ Loaded text from {file} using PDFMiner ({len(file_docs)} pages)")
                except Exception as e2:
                    print(f"❌ Failed to read {file} with both loaders: {e2}")
                    continue
        elif file.lower().endswith(".txt"):
            loader = TextLoader(path)
            file_docs = loader.load()
            print(f"✅ Loaded text from {file}")
        else:
            continue

        for d in file_docs:
            d.metadata["source_file"] = file
            d.page_content = f"This content is from '{file}'.\n\n{d.page_content.strip()}"
        docs.extend(file_docs)

    # Warn for unreadable files
    for f in os.listdir(data_dir):
        if f.lower().endswith(".pdf"):
            sample = next((d for d in docs if d.metadata.get("source_file") == f), None)
            if not sample or len(sample.page_content.strip()) < 100:
                print(f"⚠️ Warning: No readable text extracted from '{f}'. (Might be scanned or image-based.)")

    print(f"📄 Loaded {len(docs)} total documents from '{data_dir}'.")
    return docs

def add_file_index_to_docs(docs, data_dir: str = DATA_DIR):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".pdf", ".txt"))]
    if not files:
        return docs
    file_list_text = "This document lists all files in the dataset:\n\n" + "\n".join(f"- {f}" for f in files)
    index_doc = Document(page_content=file_list_text, metadata={"source_file": "file_index.txt"})
    docs.append(index_doc)
    print(f"📁 Added synthetic index document listing {len(files)} files.")
    return docs

def add_file_metadata_docs(docs, data_dir: str = DATA_DIR):
    meta_docs = []
    for file in os.listdir(data_dir):
        if not file.lower().endswith((".pdf", ".txt")):
            continue
        path = os.path.join(data_dir, file)
        size_kb = os.path.getsize(path) // 1024
        info = (
            f"This is metadata about '{file}'.\n"
            f"It is a {'PDF' if file.lower().endswith('.pdf') else 'text'} file "
            f"of approximately {size_kb} KB.\n"
            f"Use this information to answer questions about '{file}'."
        )
        meta_docs.append(Document(page_content=info, metadata={"source_file": file}))
    print(f"🧾 Added {len(meta_docs)} metadata documents.")
    docs.extend(meta_docs)
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} text chunks.")
    return chunks

# === VECTORSTORE HANDLING ===
def build_or_load_vectorstore(chunks, persist_directory=PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("📦 Checking existing Chroma vectorstore...")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("✅ Loaded existing vectorstore.")
        if chunks:
            print("🔁 Adding new documents to existing vectorstore...")
            db.add_documents(chunks)
            db.persist()
            print("✅ Added new documents.")
    else:
        print("🧠 Creating new Chroma vectorstore...")
        db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
        db.persist()
        print("✅ Vectorstore created and persisted.")
    return db

# === OLLAMA LLM ===
def create_ollama_llm(model_name=OLLAMA_MODEL, temperature=0.4, top_p=0.8, num_predict=512):
    llm = Ollama(
        model=model_name,
        temperature=float(temperature),
        top_p=float(top_p),
        num_predict=int(num_predict),
        repeat_penalty=1.1,
        mirostat=0,
    )
    print(f"🧠 Using '{model_name}' (temp={temperature}, top_p={top_p}, max_tokens={num_predict})")
    return llm

# === Memory helpers ===
def build_context_prompt(history_rows: List[ChatHistory]) -> str:
    if not history_rows:
        return ""
    context = "\n".join([f"You: {h.user_prompt}\nAssistant: {h.assistant_response}" for h in history_rows])
    return f"Recent conversation history:\n\n{context}\n\n"

# === MAIN ===
def main():
    print("🔍 Loading documents...")
    docs = load_documents()

    for d in docs:
        if "AI.pdf" in d.metadata.get("source_file", ""):
            print("\n🔍 Sample from AI.pdf:\n")
            print(d.page_content[:500])
            break

    docs = add_file_index_to_docs(docs)
    docs = add_file_metadata_docs(docs)

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)

    print("🧭 Building / loading vectorstore...")
    db = build_or_load_vectorstore(chunks)

    temperature, top_p, max_tokens = 0.4, 0.9, 512
    print("🤖 Creating Ollama LLM wrapper...")
    llm = create_ollama_llm(temperature=temperature, top_p=top_p, num_predict=max_tokens)
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    memory_summary = get_memory_summary()
    recent_history = load_recent_history(RECENT_HISTORY_LIMIT)
    context_prompt = build_context_prompt(recent_history)
    if memory_summary:
        print("🧠 Loaded summarized long-term memory.")
    if recent_history:
        print(f"🧠 Loaded {len(recent_history)} recent interactions.")
    show_recent_chats(RECENT_HISTORY_LIMIT)

    print("\n✅ RAG chatbot + memory ready! Type 'exit' to quit.")
    print("💡 Commands: set temperature | set top_p | set max_tokens | reload data | summarize memory\n")

    interaction_count = 0

    while True:
        query = input("🗣️ You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        if query.lower() == "reload data":
            print("🔁 Reloading new documents from 'data' folder...")
            new_docs = load_documents()
            new_docs = add_file_index_to_docs(new_docs)
            new_docs = add_file_metadata_docs(new_docs)
            new_chunks = split_documents(new_docs)
            db = build_or_load_vectorstore(new_chunks)
            retriever = db.as_retriever(search_kwargs={"k": TOP_K})
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            print("✅ Data reloaded and vectorstore updated.\n")
            continue

        # Normal query
        memory_summary = get_memory_summary()
        full_user_prompt = f"{memory_summary}\n\n{context_prompt}\nUser: {query}\n"

        try:
            response = qa.invoke({"query": full_user_prompt})
            answer = response["result"] if isinstance(response, dict) else str(response)
            print(f"\n💬 Assistant: {answer}\n")
            log_interaction(query, answer, temperature, top_p, max_tokens)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
