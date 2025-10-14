import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredPDFLoader




# === CONFIGURATION ===
DATA_DIR = "data"
PERSIST_DIR = "vectorstore_ollama"
OLLAMA_MODEL = "llama3"  # or whichever model you pulled into Ollama
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 6


# === DOCUMENT PROCESSING ===
def load_documents(data_dir=DATA_DIR):
    """Load PDFs and text files, embedding filename metadata into content."""
    docs = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith(".pdf"):
            loader = UnstructuredPDFLoader(path)
            file_docs = loader.load()
        elif file.endswith(".txt"):
            loader = TextLoader(path)
            file_docs = loader.load()
        else:
            continue

        for d in file_docs:
            d.metadata["source_file"] = file
            d.page_content = f"This content is from '{file}'.\n\n{d.page_content}"
        docs.extend(file_docs)

    print(f"📄 Loaded {len(docs)} documents from '{data_dir}'.")
    return docs


def add_file_index_to_docs(docs, data_dir=DATA_DIR):
    """Add a synthetic document that lists all filenames for retrieval."""
    files = [f for f in os.listdir(data_dir) if f.endswith(('.pdf', '.txt'))]
    if not files:
        return docs

    file_list_text = "This document lists all files in the dataset:\n\n"
    for f in files:
        file_list_text += f"- {f}\n"

    index_doc = Document(page_content=file_list_text, metadata={"source_file": "file_index.txt"})
    docs.append(index_doc)
    print(f"📁 Added synthetic index document listing {len(files)} files.")
    return docs


def add_file_metadata_docs(docs, data_dir=DATA_DIR):
    """Add synthetic documents describing each file’s metadata."""
    meta_docs = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith((".pdf", ".txt")):
            size_kb = os.path.getsize(path) // 1024
            info = (
                f"This is metadata about '{file}'.\n"
                f"It is a {'PDF' if file.endswith('.pdf') else 'text'} file "
                f"of approximately {size_kb} KB.\n"
                f"This file may contain historical, economic, or academic content.\n"
                f"Use this information to answer questions about '{file}'."
            )
            meta_docs.append(Document(page_content=info, metadata={"source_file": file}))
    print(f"🧾 Added {len(meta_docs)} metadata documents.")
    docs.extend(meta_docs)
    return docs


def split_documents(docs):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} text chunks.")
    return chunks


# === VECTORSTORE HANDLING ===
def build_or_load_vectorstore(chunks, persist_directory=PERSIST_DIR):
    """Create or load Chroma vectorstore with better metadata weighting."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    adjusted_docs = []
    for doc in chunks:
        src = doc.metadata.get("source_file", "unknown file")

        # Distinguish real content from metadata
        if "metadata about" in doc.page_content.lower():
            # Lower weight metadata text
            doc.page_content = (
                f"[Metadata summary about {src}] "
                f"(Use only if asked about file details like size or type.)\n{doc.page_content}"
            )
        else:
            # Encourage the model to see this as rich, summarizable content
            doc.page_content = (
                f"[Main content from {src}] "
                f"(Contains substantive information useful for summarization or Q&A.)\n{doc.page_content}"
            )

        adjusted_docs.append(doc)

    # Create or load Chroma DB
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("📦 Loading existing Chroma vectorstore...")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("🧠 Creating new Chroma vectorstore...")
        db = Chroma.from_documents(adjusted_docs, embeddings, persist_directory=persist_directory)
        db.persist()
        print("✅ Vectorstore created and persisted.")

    return db



# === LLM SETUP ===
def create_ollama_llm(model_name=OLLAMA_MODEL):
    """Create an Ollama LLM (latest LangChain schema)."""
    return Ollama(model=model_name)


# === MAIN ===
def main():
    print("🔍 Loading documents...")
    docs = load_documents()
    docs = add_file_index_to_docs(docs)
    docs = add_file_metadata_docs(docs)

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)

    print("🧭 Building / loading vectorstore...")
    db = build_or_load_vectorstore(chunks)

    print("🤖 Creating Ollama LLM wrapper...")
    llm = create_ollama_llm()

    print("🔁 Creating RetrievalQA chain (RAG)...")
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("\n✅ RAG ready! Type 'exit' to quit.\n")

    while True:
        query = input("🗣️ You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        try:
            retrieved_docs = retriever.invoke(query)
            print(f"\n🔎 Retrieved {len(retrieved_docs)} docs for query: '{query}'")
            for i, doc in enumerate(retrieved_docs[:2], 1):
                print(f"\n--- Retrieved chunk {i} ---\n{doc.page_content[:400]}\n")

            if not retrieved_docs:
                print("⚠️ No relevant context found. Using Ollama directly...\n")
                answer = llm.invoke(query)
                print("💬 Assistant:", answer, "\n")
            else:
                response = qa.invoke(
                    {"query": query},
                    config={"configurable": {"temperature": 0.1, "num_predict": 512}},
                )
                print("\n💬 Assistant:", response["result"], "\n")

                # Optional: show sources
                if "source_documents" in response:
                    sources = {doc.metadata.get("source_file", "unknown") for doc in response["source_documents"]}
                    print(f"📚 Sources: {', '.join(sources)}\n")

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
