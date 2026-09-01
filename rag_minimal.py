"""
Minimal RAG example with Streamlit + FAISS
"""
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

st.set_page_config(page_title="Minimal RAG", layout="wide")
st.title("Minimal RAG Example")

# Step 1: Create or Load Vectorstore
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vectorstore():
    if os.path.exists("faiss_index"):
        embedder = get_embeddings()
        return FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    return None

# Create vectorstore
st.header("1️⃣ Create Vectorstore")
text_input = st.text_area("Paste your text here:")

if st.button("Create FAISS Index"):
    if text_input:
        embedder = get_embeddings()
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_text(text_input)
        
        # Create vectorstore
        vectorstore = FAISS.from_texts(chunks, embedder)
        
        # Save locally
        vectorstore.save_local("faiss_index")
        st.success(f"✅ Saved {len(chunks)} chunks to ./faiss_index")
        st.cache_resource.clear()

# Query vectorstore
st.header("2️⃣ Query Vectorstore")
vectorstore = get_vectorstore()

if vectorstore:
    query = st.text_input("Enter your search query:")
    k = st.slider("Return top K results", 1, 5, 3)
    
    if query:
        # Retrieve similar documents
        docs = vectorstore.similarity_search(query, k=k)
        
        st.subheader("Results:")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Result {i}:**")
            st.write(doc.page_content)
            st.divider()
else:
    st.info("📝 Create a vectorstore first to enable queries")
