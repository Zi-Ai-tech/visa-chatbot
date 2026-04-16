from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import cohere
import os
from collections import defaultdict

chat_history = defaultdict(list)

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

print("ENV PATH:", dotenv_path)
print("FILE EXISTS:", os.path.exists(dotenv_path))

load_dotenv(dotenv_path)

print("GROQ AFTER LOAD:", os.getenv("GROQ_API_KEY"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# -------------------------
# VALIDATION (IMPORTANT)
# -------------------------
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

if not COHERE_API_KEY:
    raise ValueError("❌ COHERE_API_KEY not found in .env")

app = Flask(__name__)

# -------------------------
# LOAD MODELS (SAFE)
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "visa_vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

co = cohere.Client(COHERE_API_KEY)

# -------------------------
# BUILD BM25 INDEX
# -------------------------
docs_all = vector_store.similarity_search("visa", k=1000)
corpus = [doc.page_content for doc in docs_all]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# -------------------------
# MULTI QUERY
# -------------------------
def generate_queries(query):
    try:
        prompt = f"Generate 3 search queries for: {query}"
        response = llm.invoke(prompt).content
        queries = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
        return [query] + queries[:3]
    except:
        return [query]

# -------------------------
# HYBRID SEARCH
# -------------------------
def hybrid_search(query):
    faiss_docs = vector_store.similarity_search(query, k=5)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    bm25_docs = [docs_all[i] for i in top_indices]

    return faiss_docs + bm25_docs

# -------------------------
# RERANKING (SAFE)
# -------------------------
def rerank_docs(query, docs):
    try:
        texts = [doc.page_content for doc in docs]

        result = co.rerank(
            query=query,
            documents=texts,
            top_n=5
        )

        return [docs[r.index] for r in result.results]

    except Exception as e:
        print("Rerank failed:", e)
        return docs[:5]

# -------------------------
# API ROUTE
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_query = data.get("query")
        session_id = data.get("session_id", "default")

        # -------------------------
        # 🔥 GET CHAT HISTORY
        # -------------------------
        history = chat_history[session_id]

        # Build memory context
        memory_context = ""
        for msg in history[-6:]:  # last 6 messages
            memory_context += f"{msg['role']}: {msg['content']}\n"

        # -------------------------
        # 🔥 ENHANCE QUERY WITH MEMORY
        # -------------------------
        enhanced_query = f"""
Conversation so far:
{memory_context}

New question:
{user_query}

Rewrite the question clearly for search:
"""

        refined_query = llm.invoke(enhanced_query).content

        # -------------------------
        # 🔥 RAG PIPELINE (use refined query)
        # -------------------------
        queries = generate_queries(refined_query)

        docs = []
        for q in queries:
            docs.extend(hybrid_search(q))

        docs = rerank_docs(refined_query, docs)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a visa assistant.

Use conversation context + retrieved documents.

Conversation:
{memory_context}

Documents:
{context}

Question:
{user_query}

Answer clearly:
"""

        response = llm.invoke(prompt)

        answer = response.content

        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in docs
        ]))

        # -------------------------
        # 🔥 SAVE MEMORY
        # -------------------------
        chat_history[session_id].append({
            "role": "user",
            "content": user_query
        })

        chat_history[session_id].append({
            "role": "assistant",
            "content": answer
        })

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)