from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("Warning: HF_TOKEN not found in .env file. Rate limits may apply.")
# Check if vector store exists
if not os.path.exists("visa_vector_store"):
    print("ERROR: visa_vector_store not found!")
    print("Run create_vector_store.py first.")
    exit()

print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("visa_vector_store", embeddings, allow_dangerous_deserialization=True)

print("Checking Ollama...")
llm = OllamaLLM(model="llama3.2", temperature=0)

print("Creating RAG chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

def ask_visa_question(question):
    print(f"\n📝 Question: {question}")
    print("🤔 Thinking...")
    result = qa_chain.invoke({"query": question})
    
    print(f"\n🤖 Answer: {result['result']}")
    print(f"\n📚 Sources:")
    sources_shown = set()
    for doc in result['source_documents']:
        source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
        if source_name not in sources_shown:
            print(f"  📄 {source_name}")
            sources_shown.add(source_name)
    
    return result

if __name__ == "__main__":
    print("\n" + "="*50)
    print("RAG QUERY TEST")
    print("="*50)
    ask_visa_question("What are the requirements for a visa?")