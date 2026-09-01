import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os

# Set page config
st.set_page_config(page_title="RAG with Streamlit", layout="wide")
st.title("🔍 RAG (Retrieval-Augmented Generation) Example")

# Sidebar for configuration
st.sidebar.header("Configuration")
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)

# Initialize embeddings (cached for performance)
@st.cache_resource
def load_embeddings():
    """Load embeddings model - cached to avoid reloading"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vectorstore (cached)
@st.cache_resource
def load_vectorstore(vectorstore_path="saudi_history_faiss"):
    """Load existing vectorstore or return None"""
    embedder = load_embeddings()
    if os.path.exists(vectorstore_path):
        try:
            vectorstore = FAISS.load_local(
                vectorstore_path, 
                embedder, 
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error loading vectorstore: {e}")
            return None
    return None

# Tab 1: Upload and Process Documents
tab1, tab2 = st.tabs(["📚 Create RAG System", "🤖 Query RAG"])

with tab1:
    st.header("Step 1: Create Vectorstore")
    
    # Option 1: Upload PDF/Text files
    uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
    
    # Option 2: Paste text directly
    pasted_text = st.text_area("Or paste text directly:")
    
    if st.button("Create Vectorstore"):
        if uploaded_file or pasted_text:
            embedder = load_embeddings()
            
            # Get text content
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    documents = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    st.info("For PDF, you'll need PyPDF2: pip install PyPDF2")
                    from PyPDF2 import PdfReader
                    reader = PdfReader(uploaded_file)
                    documents = "\n".join([page.extract_text() for page in reader.pages])
            else:
                documents = pasted_text
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            docs = text_splitter.split_text(documents)
            
            st.info(f"📊 Created {len(docs)} chunks from your documents")
            
            # Create vectorstore using FAISS
            vectorstore = FAISS.from_texts(docs, embedder)
            
            # Save vectorstore locally
            vectorstore_path = "saudi_history_faiss"
            vectorstore.save_local(vectorstore_path)
            st.success(f"✅ Vectorstore saved to ./{vectorstore_path}")
            
            # Clear cache to reload vectorstore
            st.cache_resource.clear()
        else:
            st.warning("Please upload a file or paste text content")

with tab2:
    st.header("Step 2: Query Your RAG System")
    
    # Load vectorstore
    vectorstore = load_vectorstore()
    
    if vectorstore is None:
        st.warning("⚠️ No vectorstore found. Please create one first in the 'Create RAG System' tab.")
    else:
        st.success("✅ Vectorstore loaded successfully")
        
        # Query settings
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("Number of documents to retrieve (k)", 1, 10, 3)
        with col2:
            llm_model = st.selectbox(
                "LLM Model",
                ["ollama (local)", "openai", "huggingface"]
            )
        
        # Search query
        query = st.text_input("Enter your question:")
        
        if query:
            # Retrieve relevant documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(query)
            
            st.subheader("📄 Retrieved Documents")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Document {i} (Score: High)"):
                    st.write(doc.page_content)
            
            # Generate answer using RAG
            if st.button("Generate Answer with RAG"):
                st.subheader("🤖 AI Answer")
                
                if llm_model == "ollama (local)":
                    try:
                        llm = Ollama(model="llama2")
                        
                        # Create RAG chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True
                        )
                        
                        result = qa_chain(query)
                        st.write(result["result"])
                        
                    except Exception as e:
                        st.error(f"Error: {e}\n\nMake sure Ollama is running: `ollama run llama2`")
                
                elif llm_model == "openai":
                    st.info("Set your OpenAI API key first")
                    api_key = st.text_input("Enter OpenAI API key:", type="password")
                    if api_key:
                        os.environ["OPENAI_API_KEY"] = api_key
                        from langchain_openai import ChatOpenAI
                        
                        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever
                        )
                        result = qa_chain(query)
                        st.write(result["result"])

# Sidebar: Show vectorstore info
st.sidebar.header("Vectorstore Info")
vectorstore = load_vectorstore()
if vectorstore:
    st.sidebar.success("✅ Vectorstore loaded")
    # Get approximate size
    st.sidebar.write(f"📁 Location: ./saudi_history_faiss")
else:
    st.sidebar.warning("❌ No vectorstore found")

st.sidebar.markdown("---")
st.sidebar.markdown("### How RAG Works:")
st.sidebar.markdown("""
1. **Chunk Documents**: Split large documents into manageable pieces
2. **Embed**: Convert text chunks to embeddings using a model
3. **Store**: Save embeddings in a vector database (FAISS)
4. **Retrieve**: When queried, find similar chunks using vector similarity
5. **Generate**: Pass retrieved chunks to an LLM to generate answers
""")
