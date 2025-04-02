# app.py
import streamlit as st
import google.generativeai as genai
import io
import logging
from typing import List, Optional, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import docx
import pandas as pd
from PIL import Image
import faiss
import numpy as np

# --- Constants ---
AVAILABLE_GEN_MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]
DEFAULT_GEN_MODEL = AVAILABLE_GEN_MODELS[0]
EMBEDDING_MODEL = "models/embedding-001"
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150
TOP_K, HISTORY_LIMIT, API_TIMEOUT = 5, 5, 120
SUPPORTED_EXTENSIONS = ["pdf", "docx", "xlsx", "png", "jpg", "jpeg"]

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Session State Initialization ---
defaults = {
    "processed": False,
    "vector_index": None,
    "text_chunks": [],
    "messages": [],
    "selected_model": DEFAULT_GEN_MODEL,
    "analyze_images": False,
    "api_configured": False,
    "user_api_key": None,
    "processed_filenames": [],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# --- Utility Functions ---
def handle_api_error(e: Exception, context: str, model: str = "") -> None:
    """Handle API errors with user-friendly messages."""
    error_msg = str(e).lower()
    logger.error(f"{context}: {e}")
    messages = {
        "api key not valid": "Invalid or expired API key. Please check and update.",
        "permission denied": f"Permission denied for model '{model}'. Check access rights.",
        "deadline exceeded": f"Request timed out after {API_TIMEOUT}s. Try again later.",
        "resource exhausted": "API quota exceeded. Check your limits.",
    }
    for key, msg in messages.items():
        if key in error_msg:
            st.error(f"{msg} ({context})")
            if "api key" in key:
                st.session_state.api_configured = False
                st.session_state.user_api_key = None
                st.rerun()
            return
    st.error(f"API error: {e} ({context})")

def get_model(model_name: str) -> Optional[genai.GenerativeModel]:
    """Load a generative model."""
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        handle_api_error(e, f"Loading model '{model_name}'", model_name)
        return None

def describe_image(image_bytes: bytes, filename: str, model_name: str) -> Optional[str]:
    """Generate an image description."""
    model = get_model(model_name)
    if not model:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content(
            ["Describe this image in detail. Transcribe any text accurately.", img],
            request_options={"timeout": API_TIMEOUT}
        )
        return response.text.strip() if hasattr(response, 'text') else None
    except Exception as e:
        handle_api_error(e, f"Analyzing image '{filename}'", model_name)
        return None

# --- File Loaders ---
def load_pdf(file, analyze_images: bool, gen_model_name: str) -> Optional[str]:
    """Extract text and optionally images from a PDF."""
    content = []
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file.getvalue()))
        for i, page in enumerate(pdf_reader.pages):
            page_num = i + 1
            text = page.extract_text() or ""
            if text:
                content.append(f"\n--- Page {page_num} Text ---\n{text.strip()}")
            if analyze_images and st.session_state.api_configured and page.images:
                for j, img in enumerate(page.images):
                    desc = describe_image(img.data, f"{file.name} (Page {page_num}, Img {j+1})", gen_model_name)
                    if desc:
                        content.append(f"\n--- Page {page_num} Image {j+1} ---\n{desc}")
        return "\n".join(content).strip() or None
    except Exception as e:
        logger.error(f"Failed to process PDF '{file.name}': {e}")
        return None

def load_docx(file) -> Optional[str]:
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file.getvalue()))
        return "\n".join(p.text for p in doc.paragraphs if p.text).strip() or None
    except Exception as e:
        logger.error(f"Failed to process DOCX '{file.name}': {e}")
        return None

def load_excel(file) -> Optional[str]:
    """Extract text from Excel sheets."""
    try:
        sheets = pd.read_excel(io.BytesIO(file.getvalue()), sheet_name=None, engine='openpyxl')
        content = [f"\n--- Sheet: {name} ---\n{df.to_string(index=False, na_rep='NA').strip()}" 
                   for name, df in sheets.items() if not df.empty]
        return "\n".join(content).strip() or None
    except Exception as e:
        logger.error(f"Failed to process Excel '{file.name}': {e}")
        return None

def load_image(file, gen_model_name: str) -> Optional[str]:
    """Describe an image file."""
    if not st.session_state.api_configured:
        return None
    desc = describe_image(file.getvalue(), file.name, gen_model_name)
    return f"\n--- Image: {file.name} ---\n{desc}" if desc else None

SUPPORTED_MIME_TYPE_LOADERS = {
    "application/pdf": load_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": load_docx,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": load_excel,
    "image/png": load_image,
    "image/jpeg": load_image,
    "image/jpg": load_image,
}

# --- Text Processing ---
def split_text(text: str) -> List[str]:
    """Split text into chunks."""
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

@st.cache_data(show_spinner="Embedding new chunks...")
def get_embeddings_for_new_chunks(_new_chunks_tuple: Tuple[str, ...]) -> Optional[List[List[float]]]:
    """Generate embeddings for new text chunks."""
    chunks = list(_new_chunks_tuple)
    if not chunks or not st.session_state.api_configured:
        return None
    try:
        response = genai.embed_content(model=EMBEDDING_MODEL, content=chunks, task_type="retrieval_document")
        return response.get("embedding")
    except Exception as e:
        handle_api_error(e, "Embedding new chunks")
        return None

# --- RAG Pipeline ---
def embed_query(query: str) -> Optional[np.ndarray]:
    """Embed the user query."""
    try:
        response = genai.embed_content(model=EMBEDDING_MODEL, content=query, task_type="retrieval_query")
        embedding = response.get("embedding")
        return np.array(embedding, dtype="float32").reshape(1, -1) if embedding else None
    except Exception as e:
        handle_api_error(e, "Embedding query")
        return None

def retrieve_chunks(query_emb: np.ndarray, index: faiss.Index, chunks: List[str]) -> str:
    """Retrieve relevant chunks from the index."""
    if not index or index.ntotal == 0:
        return ""
    k = min(TOP_K, index.ntotal)
    distances, indices = index.search(query_emb, k)
    return "\n\n---\n\n".join(chunks[i] for i in indices[0] if 0 <= i < len(chunks))

def format_history(history: List[Dict[str, str]]) -> str:
    """Format chat history for the prompt."""
    start = max(0, len(history) - (HISTORY_LIMIT * 2))
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[start:])

def get_rag_response(query: str, index: faiss.Index, chunks: List[str], history: List[Dict[str, str]], model_name: str) -> str:
    """Generate a response using RAG."""
    if not st.session_state.api_configured:
        return "API not configured. Set your API key in the sidebar."
    if not index or not chunks or index.ntotal == 0:
        return "No documents processed. Upload files first."
    if not query:
        return "Enter a query."

    model = get_model(model_name)
    if not model:
        return f"Model '{model_name}' unavailable."

    query_emb = embed_query(query)
    if query_emb is None:  # Fixed: Explicit None check
        return "Failed to process query."

    context = retrieve_chunks(query_emb, index, chunks)
    if not context:
        return "No relevant information found."

    prompt = f"""Answer based only on this context and history:
Context:
---
{context}
---
History:
{format_history(history)}
---
Query: {query}"""
    
    try:
        response = model.generate_content(prompt, request_options={"timeout": API_TIMEOUT})
        return response.text.strip() if hasattr(response, 'text') else "No answer provided."
    except Exception as e:
        handle_api_error(e, "Generating response", model_name)
        return "Response generation failed."

# --- Streamlit UI ---
st.set_page_config(page_title="Doc Q&A", layout="wide")
st.title("💬 Document Q&A")

with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Google API Key:", type="password", value=st.session_state.get("user_api_key", ""))
    if st.button("Set API Key") and api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.api_configured = True
            st.session_state.user_api_key = api_key
            st.rerun()
        except Exception as e:
            handle_api_error(e, "API Configuration")

    st.header("⚙️ Settings")
    st.session_state.selected_model = st.selectbox("Model", AVAILABLE_GEN_MODELS, index=AVAILABLE_GEN_MODELS.index(st.session_state.selected_model), disabled=not st.session_state.api_configured)
    st.session_state.analyze_images = st.toggle("Analyze PDF Images", st.session_state.analyze_images, disabled=not st.session_state.api_configured)

    st.header("📄 Upload")
    files = st.file_uploader("Files", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True, disabled=not st.session_state.api_configured)
    if st.button("Process", disabled=not files):
        new_files = [f for f in files if f.name not in st.session_state.processed_filenames]
        if new_files:
            st.session_state.messages = []
            content = []
            for file in new_files:
                loader = SUPPORTED_MIME_TYPE_LOADERS.get(file.type)
                if loader:
                    text = loader(file, st.session_state.analyze_images, st.session_state.selected_model)
                    if text:
                        content.append(text)
                        st.session_state.processed_filenames.append(file.name)
            if content:
                chunks = split_text("\n\n".join(content))
                embeddings = get_embeddings_for_new_chunks(tuple(chunks))
                if embeddings:
                    embeddings_np = np.array(embeddings, dtype="float32")
                    if st.session_state.vector_index is None:
                        st.session_state.vector_index = faiss.IndexFlatL2(embeddings_np.shape[1])
                    st.session_state.vector_index.add(embeddings_np)
                    st.session_state.text_chunks.extend(chunks)
                    st.session_state.processed = True

    st.subheader("Processed Files")
    if st.session_state.processed_filenames:
        st.write(", ".join(st.session_state.processed_filenames))

    if st.button("Clear All"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.cache_data.clear()
        st.rerun()

# --- Chat Interface ---
if not st.session_state.api_configured:
    st.warning("Set your API key in the sidebar.")
elif not st.session_state.processed:
    st.info("Upload and process documents.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if query := st.chat_input("Ask a question:"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            response = get_rag_response(query, st.session_state.vector_index, st.session_state.text_chunks, st.session_state.messages[:-1], st.session_state.selected_model)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})