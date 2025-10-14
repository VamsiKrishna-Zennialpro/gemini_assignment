from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/AI.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages.")
print("First page preview:")
print(docs[0].page_content[:1000])
