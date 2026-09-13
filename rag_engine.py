import os
import hashlib
import pytesseract
import streamlit as st
from pdf2image import convert_from_path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\LENOVO\Downloads\Release-26.07.0-0\poppler-26.07.0\Library\bin"
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
load_dotenv()
from langchain_core.documents import Document

def load_and_chunk_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception:
        raise ValueError("Could not read this PDF. It may be corrupted or password-protected.")

    if len(pages) == 0:
        raise ValueError("This PDF appears to be empty.")

    total_pages = len(pages)

    total_text = "".join([p.page_content.strip() for p in pages])
    if len(total_text) < 20:
        raise ValueError("This PDF doesn't contain readable text. Please upload a text-based PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(pages)

    return chunks, total_pages

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(chunks, file_identifier):
    embeddings = get_embedding_model()

    file_hash = hashlib.md5(file_identifier.encode()).hexdigest()[:10]
    persist_directory = f"chroma_db_{file_hash}"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def build_knowledge_base(pdf_path, max_pages=1000):
    chunks, total_pages = load_and_chunk_pdf(pdf_path)

    if total_pages > max_pages:
        raise ValueError(f"This book has {total_pages} pages. Please upload a book with {max_pages} pages or fewer.")

    vector_store = create_vector_store(chunks, pdf_path)
    return vector_store, len(chunks), total_pages

def get_relevant_chunks(vector_store, query, k=4):
    return vector_store.similarity_search(query, k=k)

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    full_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        full_text += f"\n\n--- Page {i+1} ---\n\n{text}"
    return full_text, len(images)


def generate_answer(vector_store, query, chat_history=None):
    relevant_chunks = get_relevant_chunks(vector_store, query)

    context = "\n\n---\n\n".join([chunk.page_content for chunk in relevant_chunks])
    pages = sorted(set([chunk.metadata.get("page", 0) + 1 for chunk in relevant_chunks]))

    history_text = ""
    if chat_history:
        recent_history = chat_history[-4:]
        for msg in recent_history:
            role = "Student" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""You are a helpful study assistant. Answer the student's question using only the context from the book below.

Rules:
- Use very simple, easy-to-understand words (as if explaining to a beginner)
- Be direct and to the point — no unnecessary extra information
- Keep the answer accurate and based only on the given context
- If the question is not related to the book's content, respond with: "This question doesn't seem related to the uploaded book."
- If the student asks a follow-up question (like "explain more" or "why"), use the recent conversation below to understand what they're referring to

Recent conversation:
{history_text}

Context from the book:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, pages





