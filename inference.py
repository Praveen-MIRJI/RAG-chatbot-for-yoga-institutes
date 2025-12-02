# ---------------------------
# Imports
# ---------------------------
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------
# API KEYS (from environment variables)
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# ---------------------------
# 1. Retrieval Function (NEW QDRANT API)
# ---------------------------
def retrieve_chunks(query: str, top_k=4):
    # Convert query to embedding
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vector = resp.data[0].embedding

    # Qdrant similarity search (correct method)
    results = qdrant.query_points(
        collection_name="Institutes",
        query=query_vector,
        limit=top_k
    )

    return results.points


# ---------------------------
# 2. Build RAG Context
# ---------------------------
def build_context(results):
    context = ""
    for r in results:
        if "content" in r.payload:
            context += r.payload["content"] + "\n"
    return context


# ---------------------------
# 3. Ask the RAG System
# ---------------------------
def ask_rag(query):
    results = retrieve_chunks(query)
    context = build_context(results)

    prompt = f"""
You are a yoga institute assistant. Use ONLY the context below.

Context:
{context}

Question: {query}

Answer in one clear and helpful paragraph.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ---------------------------
# 4. Run the RAG System
# ---------------------------
query = "What is the Group Classes Subscription for Indiranagar of Athayog institute?"

print(ask_rag(query))

