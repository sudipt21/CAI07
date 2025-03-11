import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import json

# Load financial dataset (Mock data for now)
financial_docs = [
    "Company X reported a revenue of $10M in Q1 2023.",
    "Net profit for Company Y increased by 15% in 2022.",
    "Company Z's earnings per share (EPS) was $2.50 in Q4 2023."
]


def preprocess_docs(docs):
    """ Tokenize and prepare documents for BM25 and embeddings."""
    return [doc.lower().split() for doc in docs]


# Initialize BM25
bm25 = BM25Okapi(preprocess_docs(financial_docs))

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(financial_docs, convert_to_numpy=True)

# Initialize FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)


def hybrid_retrieval(query):
    """Perform BM25 + Embedding retrieval."""
    bm25_scores = bm25.get_scores(query.lower().split())
    query_embedding = embedding_model.encode([query])
    _, faiss_indices = index.search(query_embedding, k=3)

    # Combine results with simple ranking
    combined_results = {}
    for i, score in enumerate(bm25_scores):
        combined_results[i] = combined_results.get(i, 0) + score
    for i in faiss_indices[0]:
        combined_results[i] = combined_results.get(i, 0) + 1  # Simple boosting

    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    return [financial_docs[i] for i, _ in sorted_results[:3]]


# Streamlit UI
st.title("Financial Report Chatbot")
query = st.text_input("Enter a financial question:")
if query:
    results = hybrid_retrieval(query)
    st.write("### Top Retrieved Answers:")
    for res in results:
        st.write(f"- {res}")

st.write("Testing Streamlit")