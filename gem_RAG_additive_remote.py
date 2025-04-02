# app.py
import streamlit as st
import google.generativeai as genai
import io
import logging
# from dotenv import load_dotenv # Not needed
from typing import List, Optional, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import docx
import pandas as pd
from PIL import Image
import faiss
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile # For type hinting

# --- Constants ---
AVAILABLE_GEN_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.5-pro-exp-03-25",
]
DEFAULT_GEN_MODEL = AVAILABLE_GEN_MODELS[0]
EMBEDDING_MODEL = "models/embedding-001"
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150
TOP_K, HISTORY_LIMIT, API_TIMEOUT = 5, 5, 120
SUPPORTED_EXTENSIONS = ["pdf", "docx", "xlsx", "png", "jpg", "jpeg"] # Use extensions for uploader

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Initialize Session State (Defines all keys used) ---
defaults = {
    "processed": False, # True if *any* file is processed
    "vector_index": None, # Holds the FAISS index object
    "text_chunks": [],   # Holds all text chunks corresponding to the index
    "messages": [],      # Chat history
    "selected_model": DEFAULT_GEN_MODEL, # Currently selected generative model
    "analyze_images": False, # Toggle state for image analysis in PDFs
    "api_configured": False, # Flag if genai is configured successfully
    "user_api_key": None,    # Stores the user-provided API key
    "processed_filenames": [], # Tracks names of files added to the index in this session
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Utility Functions ---
def handle_api_error(e: Exception, context: str, model: str = "") -> None:
    """Handles and displays API-related errors consistently."""
    error_msg = str(e).lower()
    logger.error(f"{context}: {e}") # Log the full error

    if "api key not valid" in error_msg or ("permission denied" in error_msg and "api key" in error_msg):
        st.error(f"Error: The provided Google API Key is invalid, expired, or lacks permissions. Please check your key. ({context})")
        # Reset config status if key becomes invalid
        st.session_state.api_configured = False
        st.session_state.user_api_key = None # Clear the invalid key
        st.rerun() # Rerun to reflect change in UI immediately
    elif "permission denied" in error_msg or "could not find model" in error_msg:
        st.error(f"Error: Model '{model}' not found or you don't have permission to access it. Try selecting a different model or check API key permissions. ({context})")
    elif "deadline exceeded" in error_msg or "timeout" in error_msg:
        st.error(f"Error: The request timed out after {API_TIMEOUT} seconds. The model might be busy or the request too complex. Try again later. ({context})")
    elif "resource exhausted" in error_msg:
         st.error(f"Error: API quota exceeded. Please check your Google AI Platform quotas or wait before trying again. ({context})")
    else:
        st.error(f"An API error occurred: {e}. Check logs or try again. ({context})")

def get_model(model_name: str) -> Optional[genai.GenerativeModel]:
    """Instantiates a Gemini generative model. Assumes genai is configured."""
    if not st.session_state.api_configured:
         # This case should ideally be prevented by disabling UI elements
         logger.warning("get_model called before API was configured.")
         st.error("API is not configured. Cannot load model.")
         return None
    try:
        model = genai.GenerativeModel(model_name)
        logger.info(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        handle_api_error(e, f"Failed to load model '{model_name}'", model_name)
        return None

def describe_image(image_bytes: bytes, filename: str, model_name: str) -> Optional[str]:
    """Generates a description for an image using a generative model."""
    # Only proceed if API is configured
    if not st.session_state.api_configured:
        logger.warning(f"Skipping image description for {filename}: API not configured.")
        return None

    model = get_model(model_name)
    if not model:
        st.warning(f"Cannot describe image '{filename}' as the selected model '{model_name}' failed to load.")
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = "Describe this image in detail. If there is text, transcribe it accurately."
        logger.info(f"Requesting description for image: {filename} using {model_name}")
        response = model.generate_content(
            [prompt, img],
            request_options={"timeout": API_TIMEOUT}
        )
        description = response.text.strip() if hasattr(response, 'text') and response.text else None
        if description:
            logger.info(f"Successfully described image: {filename}")
            return description
        else:
            logger.warning(f"Image description result was empty for: {filename}")
            return None
    except Exception as e:
        handle_api_error(e, f"Image analysis failed for '{filename}'", model_name)
        return None

# --- File Loaders ---
def load_pdf(file: UploadedFile, analyze_images: bool, gen_model_name: str) -> Optional[str]:
    """Extracts text and optionally image descriptions from a PDF file."""
    content = []
    filename = file.name
    logger.info(f"Processing PDF: {filename}, Analyze Images: {analyze_images}")
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file.getvalue()))
        num_pages = len(pdf_reader.pages)
        status = st.status(f"Processing PDF '{filename}' ({num_pages} pages)...", expanded=False)

        for i, page in enumerate(pdf_reader.pages):
            page_num = i + 1
            status.update(label=f"Processing Page {page_num}/{num_pages} of '{filename}'...")
            # Extract text
            try:
                text = page.extract_text()
                if text:
                    content.append(f"\n--- Page {page_num} Text ---")
                    content.append(text.strip())
            except Exception as text_e:
                 logger.warning(f"Could not extract text from page {page_num} of {filename}: {text_e}")
                 content.append(f"\n--- Page {page_num} Text (Extraction Error) ---")

            # Extract and describe images if requested *and* API is ready
            if analyze_images and st.session_state.api_configured and page.images:
                 content.append(f"\n--- Page {page_num} Images ---")
                 for j, img in enumerate(page.images):
                     img_num = j + 1
                     img_filename_context = f"Image {img_num} on Page {page_num} ({filename})"
                     status.update(label=f"Analyzing {img_filename_context}...")
                     logger.info(f"Attempting to describe: {img_filename_context}")
                     try:
                         desc = describe_image(img.data, img_filename_context, gen_model_name)
                         if desc:
                             content.append(f"Description ({img_filename_context}):\n{desc}\n")
                         else:
                             content.append(f"Description ({img_filename_context}): [Analysis failed or returned no description]\n")
                             logger.warning(f"No description obtained for {img_filename_context}")
                     except Exception as img_e:
                        logger.error(f"Error processing image {img_num} on page {page_num} of {filename}: {img_e}")
                        content.append(f"Description ({img_filename_context}): [Error during analysis: {img_e}]\n")
            elif analyze_images and not st.session_state.api_configured:
                 logger.warning(f"Skipping image analysis on page {page_num} of {filename}: API not configured.")


        status.update(label=f"Finished processing '{filename}'.", state="complete")
        logger.info(f"Successfully processed PDF: {filename}")
        return "\n".join(content).strip() or None # Return None if totally empty
    except Exception as e:
        st.error(f"Error processing PDF '{filename}': {e}")
        logger.error(f"Failed to process PDF '{filename}': {e}", exc_info=True)
        if 'status' in locals(): status.update(label=f"Error processing '{filename}'.", state="error")
        return None

def load_docx(file: UploadedFile) -> Optional[str]:
    """Extracts text from a DOCX file."""
    filename = file.name
    logger.info(f"Processing DOCX: {filename}")
    try:
        doc = docx.Document(io.BytesIO(file.getvalue()))
        text = "\n".join(p.text for p in doc.paragraphs if p.text)
        logger.info(f"Successfully processed DOCX: {filename}")
        return text.strip() or None
    except Exception as e:
        st.error(f"Error processing DOCX '{filename}': {e}")
        logger.error(f"Failed to process DOCX '{filename}': {e}", exc_info=True)
        return None

def load_excel(file: UploadedFile) -> Optional[str]:
    """Extracts text content from all sheets of an Excel file."""
    filename = file.name
    logger.info(f"Processing Excel: {filename}")
    try:
        sheets = pd.read_excel(io.BytesIO(file.getvalue()), sheet_name=None, engine='openpyxl')
        content = []
        for name, df in sheets.items():
            sheet_content = df.to_string(index=False, na_rep='NA').strip()
            if sheet_content:
                 content.append(f"\n--- Sheet: {name} ---\n{sheet_content}")
        logger.info(f"Successfully processed Excel: {filename}")
        return "\n".join(content).strip() or None
    except Exception as e:
        st.error(f"Error processing Excel '{filename}': {e}")
        logger.error(f"Failed to process Excel '{filename}': {e}", exc_info=True)
        return None

def load_image(file: UploadedFile, gen_model_name: str) -> Optional[str]:
    """Generates a description for an image file. Requires API config."""
    filename = file.name
    logger.info(f"Processing Image file: {filename}")
    if not st.session_state.api_configured:
        logger.warning(f"Skipping image file {filename}: API not configured.")
        st.warning(f"Cannot analyze image '{filename}'. API Key needed.")
        return None
    try:
        image_bytes = file.getvalue()
        with st.spinner(f"Analyzing image '{filename}'..."):
            desc = describe_image(image_bytes, filename, gen_model_name)

        if desc:
            logger.info(f"Successfully described image file: {filename}")
            return f"\n--- Image File: {filename} ---\nDescription:\n{desc}"
        else:
            st.warning(f"Could not generate description for image '{filename}'.")
            logger.warning(f"No description obtained for image file: {filename}")
            return None
    except Exception as e:
        st.error(f"Error loading or processing image '{filename}': {e}")
        logger.error(f"Failed to load/process image file '{filename}': {e}", exc_info=True)
        return None

# Map MIME types (obtained from file.type *after* upload) to loader functions
# This is separate from the extensions used in the uploader filter
SUPPORTED_MIME_TYPE_LOADERS = {
    "application/pdf": load_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": load_docx,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": load_excel,
    "image/png": load_image,
    "image/jpeg": load_image,
    "image/jpg": load_image,
    # Add plain text, csv etc. if needed, mapping to simple loaders
    # "text/plain": lambda file, **kwargs: file.getvalue().decode("utf-8"),
}

# --- Text Processing ---
def split_text(text: str) -> List[str]:
    """Splits a long text into smaller chunks using RecursiveCharacterTextSplitter."""
    if not text:
        logger.warning("split_text called with empty input.")
        return []
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=False,
        )
        chunks = splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        logger.error(f"Failed to split text: {e}", exc_info=True)
        return []

# Caching applies to embedding *a specific batch* of new chunks
@st.cache_data(show_spinner="Generating embeddings for new chunks...")
def get_embeddings_for_new_chunks(_new_chunks_tuple: Tuple[str, ...]) -> Optional[List[List[float]]]:
    """Generates embeddings for a list of NEW text chunks. Requires API config."""
    if not st.session_state.api_configured:
         st.error("API is not configured. Cannot generate embeddings.")
         return None

    new_chunks = list(_new_chunks_tuple) # Convert back from tuple
    if not new_chunks:
        logger.warning("get_embeddings_for_new_chunks called with no texts.")
        return None

    logger.info(f"Requesting embeddings for {len(new_chunks)} new text chunks using {EMBEDDING_MODEL}.")
    try:
        valid_texts = [str(t) for t in new_chunks if t]
        if not valid_texts:
             logger.warning("No valid new text content found to embed after filtering.")
             return None

        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=valid_texts,
            task_type="retrieval_document"
        )
        embeddings = response.get("embedding")
        if embeddings and len(embeddings) == len(valid_texts):
             logger.info(f"Successfully generated {len(embeddings)} new embeddings.")
             return embeddings
        else:
             logger.error(f"New embedding generation returned unexpected result or mismatched count.")
             st.error("New embedding generation failed or returned incomplete results.")
             return None
    except Exception as e:
        handle_api_error(e, "New embedding generation failed")
        return None

# --- RAG Pipeline ---
def get_rag_response(query: str, index: faiss.Index, chunks: List[str], history: List[Dict[str, str]], model_name: str) -> str:
    """Generates a response using RAG. Requires API config."""
    if not st.session_state.api_configured:
        return "Error: API is not configured. Please provide a valid API key in the sidebar."
    if not index or not chunks:
        logger.error("RAG called with no index or chunks.")
        return "Error: No documents have been processed or the vector store is missing. Please process files first."
    if index.ntotal == 0:
         logger.error("RAG called with an empty index.")
         return "Error: The document index is empty. Please process files first."
    if not query:
        return "Please enter a query."

    model = get_model(model_name)
    if not model:
        return f"Error: The selected model '{model_name}' is currently unavailable. Please select another model or check API key."

    try:
        # 1. Embed the Query
        logger.info(f"Embedding query using {EMBEDDING_MODEL}...")
        query_emb_response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_emb_response.get("embedding")
        if not query_embedding:
            st.error("Failed to embed the query.")
            logger.error("Query embedding failed.")
            return "Error: Could not generate embedding for the query."

        query_emb_np = np.array(query_embedding, dtype="float32").reshape(1, -1)
        logger.info("Query embedded successfully.")

        # 2. Retrieve Relevant Chunks
        k = min(TOP_K, index.ntotal) # Ensure k is not larger than index size
        logger.info(f"Searching index ({index.ntotal} vectors) for {k} nearest neighbors...")
        distances, indices = index.search(query_emb_np, k)

        retrieved_chunks = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)] # Safety check for index bounds
        context = "\n\n---\n\n".join(retrieved_chunks)
        if not context:
             logger.warning("No relevant context found for the query in the vector store.")
             # Consider returning a specific message or letting the LLM handle it
             # return "Could not find relevant information in the documents for your query."

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for context.")
        # Optional: Show context in UI for debugging
        # with st.expander("Retrieved Context"):
        #     st.text(context[:1000] + "...") # Show preview

        # 3. Format History
        history_limit_turns = HISTORY_LIMIT
        history_start_index = max(0, len(history) - (history_limit_turns * 2))
        formatted_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[history_start_index:]
        )
        history_prompt_part = f"Chat History (for context):\n{formatted_history}\n\n" if formatted_history else ""

        # 4. Generate Response using LLM
        prompt = f"""You are a helpful assistant answering questions based ONLY on the provided context and chat history.
Do not use any external knowledge or prior information.
If the answer is not found within the provided context, state that clearly (e.g., "Based on the provided documents, I cannot answer that.").

Context from documents:
---
{context}
---

{history_prompt_part}User Query: {query}

Assistant Response:"""

        logger.info(f"Generating response using model {model_name}...")
        with st.spinner("Thinking..."):
            response = model.generate_content(
                prompt,
                request_options={"timeout": API_TIMEOUT}
                # generation_config=genai.types.GenerationConfig(temperature=0.7) # Optional
                )

        final_response = response.text.strip() if hasattr(response, 'text') and response.text else "The model did not provide a specific answer."
        logger.info("Response generated successfully.")
        return final_response

    except Exception as e:
        # Handle potential API errors during query embedding or final generation
        handle_api_error(e, "RAG process failed", model_name)
        return f"An error occurred during response generation. Please check logs."


# --- Streamlit UI ---
st.set_page_config(page_title="Doc Q&A (Additive)", layout="wide")
st.title(f"💬 Document Q&A ({st.session_state.selected_model if st.session_state.api_configured else 'API Key Needed'})")

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 API Configuration")

    api_key_input = st.text_input(
        "Enter your Google API Key:",
        type="password",
        key="api_key_widget",
        help="Get your key from Google AI Studio. Required for all AI features.",
        value=st.session_state.get("user_api_key", "") # Persist visually if already set
    )

    if st.button("Set API Key"):
        if api_key_input:
            try:
                # Validate by configuring and maybe a light check like list_models
                genai.configure(api_key=api_key_input)
                # genai.list_models() # Uncomment for stricter validation (optional)

                st.session_state.api_configured = True
                st.session_state.user_api_key = api_key_input
                st.success("API Key configured successfully!")
                logger.info("Gemini API configured successfully via user input.")
                st.rerun() # Rerun to enable other UI elements
            except Exception as e:
                st.session_state.api_configured = False
                st.session_state.user_api_key = None
                # Use handle_api_error for consistent messaging if possible, else fallback
                if "api key not valid" in str(e).lower():
                     handle_api_error(e, "API Key Configuration")
                else:
                     st.error(f"Failed to configure API: {e}. Key might be invalid or network issue.")
                logger.error(f"Gemini API configuration failed with user key: {e}")
        else:
            st.warning("Please enter an API key.")

    if not st.session_state.api_configured:
        st.warning("Please enter and set your Google API Key to enable app features.")

    st.divider()
    st.header("⚙️ Model & Processing Settings")

    st.session_state.selected_model = st.selectbox(
        "Select Generative Model",
        AVAILABLE_GEN_MODELS,
        index=AVAILABLE_GEN_MODELS.index(st.session_state.selected_model),
        key="model_selector",
        disabled=not st.session_state.api_configured,
        help="Choose the AI model for answering questions and analyzing images."
    )
    st.caption(f"Using: {st.session_state.selected_model if st.session_state.api_configured else 'N/A'}\nEmbedding: {EMBEDDING_MODEL}")

    st.session_state.analyze_images = st.toggle(
        "Analyze Images within PDFs?",
        value=st.session_state.analyze_images,
        key="analyze_toggle",
        disabled=not st.session_state.api_configured,
        help="Enable to describe images in PDFs (uses selected model, can be slow, increases API usage)."
    )

    st.divider()
    st.header("📄 Document Upload & Processing")

    files = st.file_uploader(
        "Upload documents or images",
        type=SUPPORTED_EXTENSIONS, # Use file extensions
        accept_multiple_files=True,
        key="file_uploader",
        disabled=not st.session_state.api_configured,
        help="Select files. You can add more later by uploading again and clicking Process."
    )

    process_button_disabled = not st.session_state.api_configured or not files
    if st.button("Process Uploaded Files", key="process_button", disabled=process_button_disabled):
        new_files_to_process = []
        if files:
            current_filenames_in_uploader = {f.name for f in files}
            processed_filenames_set = set(st.session_state.processed_filenames)
            # Identify files present in the uploader but not yet in our processed list
            new_filenames = current_filenames_in_uploader - processed_filenames_set

            if not new_filenames:
                st.info("All files currently in the uploader have already been processed in this session.")
            else:
                new_files_to_process = [f for f in files if f.name in new_filenames]
                st.info(f"Found {len(new_files_to_process)} new file(s) to process: {', '.join(new_filenames)}")
        else:
            # This case should be prevented by the disabled button logic, but good to handle
            st.warning("No files found in the uploader.")

        if new_files_to_process:
            # Clear chat history when adding new documents
            st.session_state.messages = []
            logger.info("Clearing chat history due to new file processing.")

            all_new_content = []
            processed_files_this_run = [] # Track files successfully processed *in this batch*
            with st.spinner(f"Processing {len(new_files_to_process)} new file(s)..."):
                successful_new_files = 0
                for file in new_files_to_process:
                    # Determine the correct loader based on MIME type (after upload)
                    loader_func = SUPPORTED_MIME_TYPE_LOADERS.get(file.type)
                    if loader_func:
                        logger.info(f"Loading new file '{file.name}' ({file.type}) using {loader_func.__name__}")
                        loader_args = {"file": file}
                        # Pass necessary context to loaders that need it
                        if loader_func in [load_pdf, load_image]:
                            loader_args["gen_model_name"] = st.session_state.selected_model
                        if loader_func == load_pdf:
                             loader_args["analyze_images"] = st.session_state.analyze_images

                        try:
                            extracted_text = loader_func(**loader_args)
                            if extracted_text:
                                all_new_content.append(extracted_text)
                                successful_new_files += 1
                                processed_files_this_run.append(file.name) # Mark as successful for this run
                            else:
                                st.warning(f"No content extracted from new file '{file.name}'.")
                                logger.warning(f"No content extracted from new file '{file.name}'.")
                        except Exception as load_e:
                            st.error(f"Error loading new file '{file.name}': {load_e}")
                            logger.error(f"Error loading new file '{file.name}': {load_e}", exc_info=True)
                    else:
                        st.warning(f"Unsupported file type '{file.name}' ({file.type}). Skipping.")
                        logger.warning(f"Skipping unsupported file type: {file.name} ({file.type})")

            if not all_new_content:
                st.error("No content could be extracted from the newly uploaded file(s).")
            else:
                st.info(f"Extracted content from {successful_new_files}/{len(new_files_to_process)} new files.")
                # 1. Combine & Split New Text
                new_full_text = "\n\n".join(all_new_content)
                new_chunks = split_text(new_full_text)

                if not new_chunks:
                    st.error("Failed to split the extracted text from new files.")
                else:
                     # 2. Get Embeddings for New Chunks
                     new_embeddings = get_embeddings_for_new_chunks(tuple(new_chunks))

                     if not new_embeddings:
                        st.error("Failed to generate embeddings for the new text chunks. Cannot add new files.")
                     else:
                        # 3. Add to Vector Store & State
                        try:
                            embeddings_np = np.array(new_embeddings, dtype="float32")
                            if st.session_state.vector_index is None:
                                # First time processing: Create the index
                                dimension = embeddings_np.shape[1]
                                st.session_state.vector_index = faiss.IndexFlatL2(dimension)
                                logger.info(f"Created new FAISS index, dimension={dimension}.")

                            # Add new embeddings to the existing (or newly created) index
                            st.session_state.vector_index.add(embeddings_np)
                            logger.info(f"Added {len(new_embeddings)} new vectors. Index size: {st.session_state.vector_index.ntotal}")

                            # Append new chunks and update overall processed filenames list
                            st.session_state.text_chunks.extend(new_chunks)
                            st.session_state.processed_filenames.extend(processed_files_this_run)
                            st.session_state.processed = True # Mark that *some* processing has occurred

                            st.success(f"Successfully processed and added {len(processed_files_this_run)} new file(s). Total files: {len(st.session_state.processed_filenames)}")
                            # Don't rerun here, let user see success message and processed list update below
                            # st.rerun() # Avoid rerun if possible to keep success message visible

                        except Exception as index_e:
                            st.error(f"Failed to add new data to vector store: {index_e}")
                            logger.error(f"Failed to add embeddings/chunks to state: {index_e}", exc_info=True)

    # --- Display Processed Files ---
    st.divider()
    st.subheader("Processed Files in Session:")
    if st.session_state.processed_filenames:
        with st.expander(f"{len(st.session_state.processed_filenames)} file(s) processed", expanded=False):
            # Use columns for better display if many files
            cols = st.columns(2)
            for i, filename in enumerate(st.session_state.processed_filenames):
                 cols[i%2].markdown(f"- `{filename}`")
    elif not st.session_state.api_configured:
         st.info("API not configured.")
    else:
        st.info("No files processed yet in this session.")

    st.divider()
    # --- Clear Button ---
    if st.button("Clear All Docs & Chat", key="clear_all", help="Removes all processed documents, clears chat history, and resets the index for this session."):
        # Reset all relevant state keys to their initial defaults
        keys_to_clear = ["processed", "vector_index", "text_chunks", "messages", "processed_filenames"]
        for key in keys_to_clear:
            st.session_state[key] = defaults[key] # Reset to the initial default value

        # Optionally reset the image toggle (or keep user preference)
        # st.session_state.analyze_images = defaults["analyze_images"]

        # Clear Streamlit data cache (for embeddings)
        st.cache_data.clear()
        logger.info("Cleared all documents, chat, index, and caches.")
        st.success("Cleared all session data.")
        st.rerun() # Rerun to reflect the cleared state


# --- Main Chat Interface ---
if not st.session_state.api_configured:
    st.warning("⬅️ Please provide your Google API Key in the sidebar to activate the application.")
elif not st.session_state.processed:
    st.info("⬅️ API Key set. Now, please upload and process documents using the sidebar.")
else: # API is configured AND at least one file has been processed
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if query := st.chat_input("Ask about your documents...", key="chat_input", disabled=not st.session_state.vector_index): # Disable if index isn't ready
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # Pass the current index and chunks from session state
            rag_response = get_rag_response(
                query=query,
                index=st.session_state.vector_index,
                chunks=st.session_state.text_chunks,
                history=st.session_state.messages[:-1], # History before this query
                model_name=st.session_state.selected_model
            )
            st.markdown(rag_response)
            st.session_state.messages.append({"role": "assistant", "content": rag_response})

# --- Requirements Comment ---
# To run this app, you need a requirements.txt file with:
# streamlit
# google-generativeai
# langchain # (specifically for text splitter)
# pypdf
# python-docx
# pandas
# openpyxl # Needed by pandas for Excel files
# Pillow # For image handling
# faiss-cpu # or faiss-gpu if you have CUDA setup
# numpy
#
# Install using: pip install -r requirements.txt