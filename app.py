import streamlit as st
import ollama
import asyncio
import threading
import json
import os
import time
from queue import Queue
import sys
import faiss
from io import BytesIO
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


# ============= RAG FUNCTIONS =============
def process_pdf_to_vectorstore(pdf_file):
    """Traiter un PDF et créer un vector store"""
    try:
        pdf_reader = PdfReader(pdf_file)
        documents = ""
        for page in pdf_reader.pages:
            documents += page.extract_text()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(documents)

        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        dimension = len(hf_embeddings.embed_query("sample text"))

        index = faiss.IndexFlatL2(dimension)
        vector_store = FAISS(
            embedding_function=hf_embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )
        vector_store.add_texts(texts)
        return vector_store, hf_embeddings
    except Exception as e:
        st.error(f"Erreur lors du traitement du PDF: {e}")
        return None, None


def retrieve_from_rag(query: str, vector_store, k: int = 3) -> str:
    """Récupérer les documents pertinents du RAG"""
    if vector_store is None:
        return ""
    
    try:
        results = vector_store.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        print(f"DEBUG: Erreur RAG: {e}", flush=True)
        return ""





st.title("Summary Bot")
st.markdown("Je peux vous résumer vos fichiers PDF")
st.markdown("I can summarize your PDF files")


# ============= SIDEBAR =============
with st.sidebar:
    st.header("Configuration")
    
    ollama_model_name = st.selectbox(
        "Modèle Ollama",
        ["mistral:7b-instruct", "qwen3:latest", "incept5/llama3.1-claude", "deepseek-r1:latest"]
    )
    
    st.divider()
    st.subheader("📚 RAG - Ajouter des documents")
    
    use_rag = st.checkbox("Activer le RAG", value=False)
    
    if use_rag:
        uploaded_files = st.file_uploader(
            "Uploadez des PDF D&D",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Traiter le PDF"):
                with st.spinner("Traitement du PDF..."):
                    for pdf_file in uploaded_files:
                        vector_store, embeddings = process_pdf_to_vectorstore(pdf_file)
                        if vector_store:
                            st.session_state[f"vector_store_{pdf_file.name}"] = vector_store
                            st.success(f"✅ {pdf_file.name} traité!")





def generate_response(input_text):
    """Générer une réponse avec Ollama, enrichie par le RAG"""
    print(f"DEBUG: generate_response called with: {input_text[:50]}", flush=True)

    try:
        # Récupérer le contexte RAG s'il est activé
        rag_context = ""
        if st.session_state.get("use_rag", False):
            for key in st.session_state:
                if key.startswith("vector_store_"):
                    vector_store = st.session_state[key]
                    context = retrieve_from_rag(input_text, vector_store, k=3)
                    if context:
                        rag_context += f"\n{context}\n"
        

        
        system_prompt = f"""Tu es un assistant qui permet aux utilisateurs de résumer leur fichier PDF
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        response = ollama.chat(model=ollama_model_name, messages=messages)
        content = response.message.content.strip()
        
        print(f"DEBUG: Model response: {content[:200]}", flush=True)

        
    
    except Exception as e:
        print(f"DEBUG: Exception in generate_response: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return f"❌ Erreur Ollama: {str(e)}"


# ============= SESSION STATE =============
if "history_rules" not in st.session_state:
    st.session_state.history_rules = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False


def save_feedback(index):
    st.session_state.history_rules[index]["feedback"] = st.session_state[f"feedback_{index}"]


# ============= CHAT DISPLAY =============
for i, message in enumerate(st.session_state.history_rules):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history_rules.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("⏳ Génération en cours..."):
            response = generate_response(prompt)
        st.write(response)
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history_rules)}",
            on_change=save_feedback,
            args=[len(st.session_state.history_rules)],
        )
    st.session_state.history_rules.append({"role": "assistant", "content": response})