from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

loader = DirectoryLoader(
    "./visa_docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

print(f"Loaded {len(documents)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

enhanced_chunks = []

for doc in documents:
    chunks = text_splitter.split_documents([doc])
    
    for chunk in chunks:
        text = chunk.page_content.lower()
        source = chunk.metadata.get("source", "").lower()
        
        # Basic metadata extraction
        country = "unknown"
        if "canada" in text or "canada" in source:
            country = "canada"
        elif "uk" in text or "united kingdom" in text:
            country = "uk"
        
        visa_type = "unknown"
        if "student" in text or "study" in text:
            visa_type = "study"
        elif "work" in text:
            visa_type = "work"
        elif "tourist" in text or "visit" in text:
            visa_type = "tourist"
        
        chunk.metadata.update({
            "country": country,
            "visa_type": visa_type,
            "source": source
        })
        
        enhanced_chunks.append(chunk)

print(f"Created {len(enhanced_chunks)} enhanced chunks")

with open("chunks.pkl", "wb") as f:
    pickle.dump(enhanced_chunks, f)