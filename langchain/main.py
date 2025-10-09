import os
import tempfile
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredPowerPointLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, "Gemini is a multimodal large language model developed by Google DeepMind. It can handle text, images, and reasoning tasks.")
pdf.output("sample.pdf")

# ------------------------------------------------------------
# 1. Load Environment Variables
# ------------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

# ------------------------------------------------------------
# 2. Initialize Gemini Models
# ------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ------------------------------------------------------------
# 3. Document Loader Utility
# ------------------------------------------------------------
def load_document(file_path: str):
    """Load different document types (PDF, CSV, PPTX)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

# ------------------------------------------------------------
# 4. Index Documents into FAISS Vector Store
# ------------------------------------------------------------
def create_faiss_index(file_paths):
    """Create a FAISS vector store from given files."""
    all_docs = []
    for path in file_paths:
        docs = load_document(path)
        all_docs.extend(docs)
        print(f" Loaded {len(docs)} chunks from {path}")

    print("🔹 Generating embeddings and creating FAISS index...")
    vectorstore = FAISS.from_documents(all_docs, embedding=embeddings)
    print(" Vector store created successfully.")
    return vectorstore

# ------------------------------------------------------------
# 5. Prompt Template for Querying
# ------------------------------------------------------------
def build_prompt():
    """Return a custom prompt template for Gemini retrieval."""
    prompt_template = """
    You are a knowledgeable assistant. Use the following context to answer the user's question accurately.
    If the answer cannot be found in the context, say: "I don’t have enough information in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ------------------------------------------------------------
# 6. Create RAG Chain (Retrieval Augmented Generation)
# ------------------------------------------------------------
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = build_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

# ------------------------------------------------------------
# 7. Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example document paths (replace with your files)
    sample_files = [
        "sample.pdf",
        "data.csv",
        "presentation.pptx"
    ]

    # Ensure files exist
    sample_files = [f for f in sample_files if os.path.exists(f)]
    if not sample_files:
        print(" No input files found. Please place PDF, CSV, or PPTX files in this directory.")
        exit()

    # Step 1: Create FAISS index
    vectorstore = create_faiss_index(sample_files)

    # Step 2: Create QA chain
    qa_chain = create_rag_chain(vectorstore)

    # Step 3: Ask a question
    query = "Summarize the main topics covered in the uploaded documents."
    print("\n Query:", query)

    result = qa_chain.invoke({"query": query})
    print("\n Answer:", result["result"])

    # Optional: print source metadata
    print("\n Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata)
