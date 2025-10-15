# main_memory_rag.py
import os
import datetime
import time
from typing import List

# LangChain / loaders / vectorstore / embeddings / llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
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
OLLAMA_MODEL = "llama3"                      # change to your Ollama model name
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 6

# PostgreSQL connection (update to your credentials)
# Examples:
# "postgresql+psycopg2://user:password@localhost:5432/dbname"
# "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"

# Memory / summarization params
RECENT_HISTORY_LIMIT = 5        # short-term memory size (recent turns)
SUMMARY_INTERVAL = 10           # create/update long-term summary every N interactions

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

# Create engine & session (wrapped in try/except to avoid crashing if DB down)
Session = None
try:
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    print("✅ Connected to PostgreSQL.")
except Exception as e:
    print(f"❌ PostgreSQL connection failed: {e}\nProceeding without DB logging/memory (Session=None).")

# === DB helpers ===
def log_interaction(prompt: str, response: str, temperature: float, top_p: float, max_tokens: int):
    """Store each chat interaction in PostgreSQL (if available)."""
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
        # no need to flush
    except Exception as e:
        print(f"⚠️ Failed to log to PostgreSQL: {e}")
        session.rollback()
    finally:
        session.close()

def load_recent_history(limit: int = RECENT_HISTORY_LIMIT) -> List[ChatHistory]:
    """Return last `limit` chat history entries in chronological order."""
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
    """Fetch the latest long-term memory summary (text) or empty string."""
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
def load_documents(data_dir: str = DATA_DIR):
    """Load PDFs and text files. Uses UnstructuredPDFLoader which tries OCR if needed."""
    docs = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.lower().endswith(".pdf"):
            loader = UnstructuredPDFLoader(path)
            file_docs = loader.load()
        elif file.lower().endswith(".txt"):
            loader = TextLoader(path)
            file_docs = loader.load()
        else:
            continue

        for d in file_docs:
            # Attach filename metadata and prefix page content with a marker
            d.metadata["source_file"] = file
            d.page_content = f"This content is from '{file}'.\n\n{d.page_content}"
        docs.extend(file_docs)

    print(f"📄 Loaded {len(docs)} document pages/segments from '{data_dir}'.")
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
    adjusted_docs = []
    for doc in chunks:
        src = doc.metadata.get("source_file", "unknown file")
        if "metadata about" in doc.page_content.lower():
            doc.page_content = (
                f"[Metadata summary about {src}] "
                f"(Use only if asked about file details like size or type.)\n{doc.page_content}"
            )
        else:
            doc.page_content = (
                f"[Main content from {src}] "
                f"(Contains information for summarization or Q&A.)\n{doc.page_content}"
            )
        adjusted_docs.append(doc)

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("📦 Loading existing Chroma vectorstore...")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("🧠 Creating new Chroma vectorstore...")
        db = Chroma.from_documents(adjusted_docs, embeddings, persist_directory=persist_directory)
        db.persist()
        print("✅ Vectorstore created and persisted.")
    return db

# === OLLAMA LLM ===
def create_ollama_llm(model_name=OLLAMA_MODEL, temperature=0.4, top_p=0.9, num_predict=512):
    # enforce types to satisfy pydantic validation
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
    # load and index documents
    print("🔍 Loading documents...")
    docs = load_documents()
    docs = add_file_index_to_docs(docs)
    docs = add_file_metadata_docs(docs)

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)

    print("🧭 Building / loading vectorstore...")
    db = build_or_load_vectorstore(chunks)

    # defaults
    temperature = 0.4
    top_p = 0.9
    max_tokens = 512

    print("🤖 Creating Ollama LLM wrapper...")
    llm = create_ollama_llm(temperature=temperature, top_p=top_p, num_predict=max_tokens)

    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # load memory
    memory_summary = get_memory_summary()
    recent_history = load_recent_history(RECENT_HISTORY_LIMIT)
    context_prompt = build_context_prompt(recent_history)
    if memory_summary:
        print("🧠 Loaded summarized long-term memory.")
    if recent_history:
        print(f"🧠 Loaded {len(recent_history)} recent interactions.")

    # show recent chats brief
    show_recent_chats(RECENT_HISTORY_LIMIT)

    print("\n✅ RAG chatbot + memory ready! Type 'exit' to quit.")
    print("💡 Commands: set temperature 0.8 | set top_p 0.7 | set max_tokens 300 | summarize memory\n")

    interaction_count = 0  # total interactions (used for periodic summarization)

    while True:
        query = input("🗣️ You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        # dynamic settings
        if query.lower().startswith("set temperature"):
            try:
                temperature = float(query.split()[-1])
                llm = create_ollama_llm(temperature=temperature, top_p=top_p, num_predict=max_tokens)
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                print(f"✅ Temperature updated to {temperature}\n")
            except Exception:
                print("⚠️ Invalid value. Example: set temperature 0.7\n")
            continue

        if query.lower().startswith("set top_p"):
            try:
                top_p = float(query.split()[-1])
                llm = create_ollama_llm(temperature=temperature, top_p=top_p, num_predict=max_tokens)
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                print(f"✅ top_p updated to {top_p}\n")
            except Exception:
                print("⚠️ Invalid value. Example: set top_p 0.85\n")
            continue

        if query.lower().startswith("set max_tokens"):
            try:
                max_tokens = int(query.split()[-1])
                llm = create_ollama_llm(temperature=temperature, top_p=top_p, num_predict=max_tokens)
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                print(f"✅ max_tokens updated to {max_tokens}\n")
            except Exception:
                print("⚠️ Invalid value. Example: set max_tokens 256\n")
            continue

        if query.lower() == "summarize memory":
            # manual trigger to regenerate long-term summary
            print("🔁 Regenerating long-term memory summary...")
            # load all history
            if not Session:
                print("⚠️ No DB session; cannot summarize history.")
                continue
            session = Session()
            try:
                chats = session.query(ChatHistory).order_by(ChatHistory.timestamp.asc()).all()
                session.close()
            except Exception as e:
                print(f"⚠️ Failed to load history for summary: {e}")
                session.close()
                continue

            if not chats:
                print("⚠️ No chat history to summarize.")
                continue

            full_history = "\n".join([f"You: {c.user_prompt}\nAssistant: {c.assistant_response}" for c in chats])
            summary_prompt = (
                "Summarize the following conversation history into a short, concise memory "
                "listing key topics, facts, preferences, and stable user info to remember:\n\n"
                + full_history
            )

            # use llm to summarize directly (no retrieval)
            try:
                raw = llm.invoke(summary_prompt)
                if isinstance(raw, dict) and "result" in raw:
                    summary_text = raw["result"]
                else:
                    summary_text = str(raw)
                save_memory_summary(summary_text)
            except Exception as e:
                print(f"⚠️ Summarization failed: {e}")
            continue

        # Compose prompt: memory_summary + recent context + user query
        memory_summary = get_memory_summary()
        full_user_prompt = f"{memory_summary}\n\n{context_prompt}\nUser: {query}\n"

        try:
            # Use retrieval-enhanced QA with the composed prompt
            response = qa.invoke({"query": full_user_prompt})
            # response may be dict with 'result'
            if isinstance(response, dict) and "result" in response:
                answer = response["result"]
            else:
                answer = str(response)
            print(f"\n💬 Assistant: {answer}\n")

            # log and update context
            log_interaction(query, answer, temperature, top_p, max_tokens)
            context_prompt += f"You: {query}\nAssistant: {answer}\n"
            interaction_count += 1

            # Periodic automatic summarization
            if interaction_count % SUMMARY_INTERVAL == 0:
                print("🧩 Summarizing long-term memory (automatic)...")
                if not Session:
                    print("⚠️ No DB session; skipping automatic summarization.")
                    continue
                session = Session()
                try:
                    chats = session.query(ChatHistory).order_by(ChatHistory.timestamp.asc()).all()
                    session.close()
                    if chats:
                        full_history = "\n".join([f"You: {c.user_prompt}\nAssistant: {c.assistant_response}" for c in chats])
                        summary_prompt = (
                            "Summarize the following conversation history into a short, concise memory "
                            "listing key topics, facts, preferences, and stable user info to remember:\n\n"
                            + full_history
                        )
                        # use llm (no retrieval)
                        raw = llm.invoke(summary_prompt)
                        if isinstance(raw, dict) and "result" in raw:
                            summary_text = raw["result"]
                        else:
                            summary_text = str(raw)
                        save_memory_summary(summary_text)
                        # refresh memory_summary variable
                        memory_summary = summary_text
                        print("🧠 Long-term memory updated (automatic).")
                except Exception as e:
                    print(f"⚠️ Automatic summarization failed: {e}")

        except Exception as e:
            print(f"\n❌ Error during QA/invoke: {e}\n")
            # fallback: try direct llm generation without retrieval context
            try:
                raw = llm.invoke(query)
                if isinstance(raw, dict) and "result" in raw:
                    fallback_answer = raw["result"]
                else:
                    fallback_answer = str(raw)
                print(f"\n💬 Assistant (fallback): {fallback_answer}\n")
                log_interaction(query, fallback_answer, temperature, top_p, max_tokens)
                context_prompt += f"You: {query}\nAssistant: {fallback_answer}\n"
                interaction_count += 1
            except Exception as e2:
                print(f"⚠️ Fallback generation failed: {e2}")

if __name__ == "__main__":
    main()
