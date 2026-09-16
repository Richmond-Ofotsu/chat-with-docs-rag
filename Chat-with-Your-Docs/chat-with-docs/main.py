from pathlib import Path
import streamlit as st

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- 1. Page Configuration ---
st.set_page_config(page_title="Doc Q&A (Local Ollama)", page_icon="🤖")
st.title("🤖 Chat with your Document")
st.caption("Powered locally by Ollama (Qwen 3) + Chroma")


# --- 2. Build and Cache the RAG Chain ---
# @st.cache_resource ensures the PDF is loaded & embedded ONLY ONCE, not on every user question!
@st.cache_resource(show_spinner="Loading document and initializing embeddings...")
def initialize_rag_chain():
    # 1. Initialize LLM
    llm = ChatOllama(model="qwen3:1.7b")

    # 2. Load PDF
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "documents" / "sample.pdf"

    if not pdf_path.exists():
        st.error(f"Could not find PDF at: {pdf_path}")
        st.stop()

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # 3. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Embeddings
    embeddings = OllamaEmbeddings(model="qwen3-embedding")

    # 5. Vector Store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # 6. Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 7. Prompt
    prompt = ChatPromptTemplate.from_template("""
Answer the user's question based only on the following context.

Context:
{context}

Question:
{input}

Answer:
""")

    # 8. Document Chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 9. Retrieval Chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

# Load the chain
rag_chain = initialize_rag_chain()


# --- 3. Chat Session State ---
# Streamlit wipes variables on rerun, so we store messages in st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 4. Render Previous Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If this assistant message had sources, render them in an expander
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources"):
                for src in msg["sources"]:
                    st.caption(f"**Page {src.metadata.get('page', 'Unknown')}**")
                    st.text(src.page_content)


# --- 5. Handle User Input ---
if user_query := st.chat_input("Ask a question about the document..."):
    
    # 1. Display user's question and append to session
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Get answer from your RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating answer..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            sources = response.get("context", [])

            st.markdown(answer)

            # Display source snippets
            if sources:
                with st.expander("📚 View Sources"):
                    for doc in sources:
                        st.caption(f"**Page {doc.metadata.get('page', 'Unknown')}**")
                        st.text(doc.page_content)

    # 3. Append assistant's answer and sources to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })