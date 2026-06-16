import os
import hashlib
import tempfile
import warnings
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# Load environment variables from .env when running from the app folder.
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None

if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

if "processing_error" not in st.session_state:
    st.session_state.processing_error = None


def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it in .streamlit/secrets.toml or export it as an environment variable."
        )

    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=groq_api_key,
    )


def get_file_id(file_bytes, file_name):
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return f"{file_name}_{file_hash}"


def build_vectorstore_from_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        temp_pdf_path = tmp_file.name

    try:
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No content could be loaded from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            raise ValueError("No text chunks were created from the PDF.")

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore

    finally:
        try:
            os.remove(temp_pdf_path)
        except Exception:
            pass


def auto_process_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    current_file_id = get_file_id(file_bytes, uploaded_file.name)

    if st.session_state.processed_file_id == current_file_id:
        return

    with st.spinner("Processing PDF automatically..."):
        vectorstore = build_vectorstore_from_pdf(file_bytes)
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        st.session_state.processed_file_id = current_file_id
        st.session_state.processed_file_name = uploaded_file.name
        st.session_state.processing_error = None
        st.session_state.messages = []


def get_context(query):
    if st.session_state.retriever is None:
        return ""

    docs = st.session_state.retriever.invoke(query)

    if not docs:
        return ""

    context_parts = []
    for doc in docs:
        if hasattr(doc, "page_content") and doc.page_content:
            context_parts.append(doc.page_content)

    return "\n\n".join(context_parts)


uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is None:
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.processed_file_id = None
    st.session_state.processed_file_name = None
    st.session_state.processing_error = None
else:
    try:
        auto_process_pdf(uploaded_file)
        if st.session_state.processed_file_name:
            st.success(f"PDF auto-processed: {st.session_state.processed_file_name}")
    except Exception as e:
        st.session_state.processing_error = str(e)
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.error(f"Error while processing PDF: {st.session_state.processing_error}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about the uploaded PDF...")

if prompt:
    if st.session_state.retriever is None:
        st.error("Please upload a PDF first.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            with st.spinner("Generating answer..."):
                context = get_context(prompt)

                if not context:
                    response = (
                        "I could not find relevant information in the uploaded PDF."
                    )
                else:
                    final_prompt = f"""
You are a helpful AI assistant.
Answer the user's question only from the context below.
Keep the answer clear and concise.
If the answer is not available in the context, say:
"The answer is not available in the uploaded document."

Context:
{context}

Question:
{prompt}

Answer:
"""
                    llm = get_llm()
                    result = llm.invoke(final_prompt)
                    response = result.content

        except Exception as e:
            response = f"Error: {str(e)}"

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
