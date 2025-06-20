# Teachings of Lord Chaitanya Chatbot – DOCX Version (Hinglish, No OpenAI)

import streamlit as st
from docx import Document
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

# ---------------------- DOCX Processing ----------------------

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        st.error(f"Failed to read DOCX file: {e}")
        return ""

def split_text(text, max_chunk_size=1000):
    sentences = re.split(r'[\n\.!?]', text)
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_chunk_size:
            chunk += sentence.strip() + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence.strip() + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# ---------------------- Embedding & Search ----------------------

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def build_index(chunks, model):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, chunks

def search_answers(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Lord Chaitanya Chatbot (DOCX)")
st.title("📖 Teachings of Lord Chaitanya – Hinglish Q&A Chatbot")
st.markdown("Upload the DOCX file and ask questions in Hinglish. Get answers directly from the book!")

uploaded_file = st.file_uploader("📥 Upload 'Teachings of Lord Chaitanya' .docx file:", type="docx")

if uploaded_file is not None:
    with st.spinner("Processing DOCX..."):
        text = extract_text_from_docx(uploaded_file)
        if not text.strip():
            st.error("No readable text found in DOCX file.")
        else:
            chunks = split_text(text)
            model = load_model()
            index, embeddings, chunk_list = build_index(chunks, model)
            st.success("Chatbot ready! Ask your question below.")

            user_query = st.text_input("💬 Ask your question (Hinglish allowed):")

            if user_query:
                answers = search_answers(user_query, model, index, chunk_list)
                st.markdown("### 📜 Answer(s):")
                for i, ans in enumerate(answers, 1):
                    st.markdown(f"**{i}.** {ans}")
else:
    st.info("Please upload the DOCX file to begin.")
