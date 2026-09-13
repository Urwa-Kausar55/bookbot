import streamlit as st
import tempfile
import os
from rag_engine import build_knowledge_base, generate_answer

st.set_page_config(
    page_title="BookBot",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem 3rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    h1 {
        color: #2c3e50;
    }
    .stAlert {
        border-radius: 10px;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "book_loaded" not in st.session_state:
    st.session_state.book_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📚 BookBot")
st.caption("Your personal study assistant — upload a book, ask anything about it.")

with st.sidebar:
    st.markdown("### 📤 Upload a Book")
    st.markdown("Upload a text-based PDF to get started.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

    if st.button("🚀 Process Book", type="primary"):
        if uploaded_file:
            with st.spinner("📖 Reading and indexing your book..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    vector_store, num_chunks, total_pages = build_knowledge_base(tmp_path)
                    st.session_state.vector_store = vector_store
                    st.session_state.book_loaded = True
                    st.session_state.chat_history = []
                    st.success(f"✅ Ready! Indexed {total_pages} pages.")

                    os.unlink(tmp_path)
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please upload a PDF file first.")

    if st.session_state.book_loaded:
        st.markdown("---")
        if st.button("🔄 Upload a New Book"):
            st.session_state.vector_store = None
            st.session_state.book_loaded = False
            st.session_state.chat_history = []
            st.rerun()

if st.session_state.book_loaded:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "pages" in msg:
                with st.expander("📄 Pages Referenced"):
                    st.write(", ".join([str(p) for p in msg["pages"]]))

    user_question = st.chat_input("Ask a question about the book...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, pages = generate_answer(st.session_state.vector_store, user_question, st.session_state.chat_history)
                st.markdown(answer)
                with st.expander("📄 Pages Referenced"):
                    st.write(", ".join([str(p) for p in pages]))

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "pages": pages
        })
else:
    st.info("👈 Upload a PDF book in the sidebar to get started.")