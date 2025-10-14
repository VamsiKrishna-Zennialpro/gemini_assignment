# main.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader


from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# 1️⃣ Load your local data
def load_documents():
    docs = []
    data_dir = "data"
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs

# 2️⃣ Split into smaller chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

# 3️⃣ Create embeddings (offline)
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="vectorstore")
    db.persist()
    return db

# 4️⃣ Load a local model for generation
def load_local_model():
    gen_pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # change model if needed
        torch_dtype="auto",
        device_map="auto",
        max_new_tokens=300,
        temperature=0.2,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

# 5️⃣ Build RAG chain
def create_qa_chain(db, llm):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def main():
    print("🔍 Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} docs")

    print("🧠 Splitting & embedding...")
    chunks = split_documents(docs)
    db = create_vectorstore(chunks)
    print("✅ Vector DB ready!")

    llm = load_local_model()
    qa_chain = create_qa_chain(db, llm)

    print("\n🤖 Offline RAG Chatbot ready! Type 'exit' to quit.\n")
    while True:
        query = input("🗣️ Ask something: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.run(query)
        print("\n💬", answer, "\n")

if __name__ == "__main__":
    main()
