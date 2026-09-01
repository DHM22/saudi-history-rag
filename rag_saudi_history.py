"""
RAG Streamlit App - History of Saudi Arabia
Based on the notebook RAG_History_of_Saudi_Arabia.ipynb

This app loads Wikipedia articles about Saudi Arabia's history,
vectorizes them with FAISS, and answers questions using an LLM.
"""

import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from bs4 import SoupStrainer
import time

# Set page config
st.set_page_config(
    page_title="Saudi Arabia History RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏛️ History of Saudi Arabia - RAG Assistant")
st.markdown("*Ask questions about Saudi Arabia's history using Retrieved-Augmented Generation*")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

# OpenAI API Key
api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# K value for retrieval
k_docs = st.sidebar.slider("Number of documents to retrieve (k)", 1, 10, 6)

# Cache control
if st.sidebar.button("🔄 Clear Cache"):
    st.cache_resource.clear()
    st.success("Cache cleared!")

# --- Data Source ---
SOURCES = {
    "History of Saudi Arabia": "https://en.wikipedia.org/wiki/History_of_Saudi_Arabia",
    "First Saudi State":       "https://en.wikipedia.org/wiki/First_Saudi_state",
    "Second Saudi State":      "https://en.wikipedia.org/wiki/Second_Saudi_state",
    "Unification (1902-1932)": "https://en.wikipedia.org/wiki/Unification_of_Saudi_Arabia",
    "Ibn Saud (founder)":      "https://en.wikipedia.org/wiki/Ibn_Saud",
}

# --- Cache Resources ---
@st.cache_resource
def load_documents():
    """Load documents from Wikipedia with caching"""
    st.info("📥 Loading documents from Wikipedia...")
    
    os.environ["USER_AGENT"] = "WCD-RAG-Exercise/1.0 (classroom demo)"
    
    # Wikipedia parser - extract only article content
    article_only = {"parse_only": SoupStrainer("div", {"class": "mw-parser-output"})}
    
    docs = {}
    for name, url in SOURCES.items():
        try:
            loader = WebBaseLoader(url, bs_kwargs=article_only)
            loaded = loader.load()
            
            # Fallback: if strainer matched nothing, load whole page
            if not loaded or len(loaded[0].page_content.strip()) < 500:
                loaded = WebBaseLoader(url).load()
            
            for d in loaded:
                d.metadata["source_name"] = name
            docs[name] = loaded
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
    
    return docs

@st.cache_resource
def chunk_documents(docs):
    """Split documents into chunks"""
    st.info("✂️ Chunking documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
    )
    
    chunks = {name: text_splitter.split_documents(d) for name, d in docs.items()}
    return chunks

@st.cache_resource
def create_vectorstore(chunks, api_key):
    """Create FAISS vectorstore from chunks"""
    if not api_key:
        st.error("⚠️ Please provide OpenAI API key in sidebar")
        return None
    
    st.info("🧮 Creating embeddings and FAISS index...")
    
    # Initialize embeddings
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    # Build index from first source
    names = list(chunks.keys())
    first = names[0]
    
    with st.spinner(f"Embedding '{first}'..."):
        vectorstore = FAISS.from_documents(chunks[first], embedder)
    
    # Add remaining sources
    for name in names[1:]:
        with st.spinner(f"Adding '{name}'..."):
            vectorstore.add_documents(chunks[name])
    
    st.success(f"✅ Vectorstore ready: {vectorstore.index.ntotal} vectors")
    return vectorstore, embedder

@st.cache_resource
def get_qa_chain(_vectorstore, api_key, k):
    """Create RAG chain - simple approach without RetrievalQA"""
    if not api_key or _vectorstore is None:
        return None
    
    # Retriever
    retriever = _vectorstore.as_retriever(search_kwargs={"k": k})
    
    # LLM
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.0,
        api_key=api_key
    )
    
    # Custom prompt that forces RAG behavior
    PROMPT = PromptTemplate(
        template="""You are a history assistant answering questions about the history of Saudi Arabia.
Use ONLY the context below. If the context does not contain the answer, say
"The retrieved sources do not cover this" instead of guessing.
Mention dates and names precisely when the context provides them.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )
    
    # Simple RAG chain using LLMChain
    class SimpleRAGChain:
        def __init__(self, llm, retriever, prompt):
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt
        
        def invoke(self, query_dict):
            query = query_dict.get("query")
            
            # Retrieve documents - use invoke() for newer LangChain versions
            docs = self.retriever.invoke(query)
            
            # Format context from retrieved docs
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt_text = self.prompt.format(context=context, question=query)
            
            # Get answer from LLM
            answer = self.llm.invoke(prompt_text)
            
            return {
                "result": answer.content if hasattr(answer, 'content') else str(answer),
                "source_documents": docs
            }
    
    return SimpleRAGChain(llm, retriever, PROMPT)

# --- Main App ---

# Check if API key is provided
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to begin")
else:
    # Load and process documents
    try:
        docs = load_documents()
        chunks = chunk_documents(docs)
        vectorstore, embedder = create_vectorstore(chunks, api_key)
        
        if vectorstore:
            qa_chain = get_qa_chain(vectorstore, api_key, k_docs)
            
            # --- Tabs for different features ---
            tab1, tab2, tab3 = st.tabs(["🤖 Ask Questions", "🔍 Explore Documents", "📊 Stats"])
            
            with tab1:
                st.header("Ask Questions")
                
                # Question input
                question = st.text_input(
                    "Enter your question about Saudi Arabia's history:",
                    placeholder="e.g., Why did the First Saudi State fall?"
                )
                
                if question:
                    with st.spinner("🔄 Searching and generating answer..."):
                        response = qa_chain.invoke({"query": question})
                    
                    # Display answer
                    st.subheader("📝 Answer")
                    st.write(response["result"])
                    
                    # Display sources
                    st.subheader("📚 Sources")
                    source_names = dict.fromkeys(
                        d.metadata["source_name"] for d in response["source_documents"]
                    )
                    for source in source_names:
                        st.markdown(f"- **{source}**")
                    
                    # Show retrieved chunks (expandable)
                    with st.expander("📄 View Retrieved Text"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            st.markdown(f"**[{i}] {doc.metadata['source_name']}**")
                            st.write(doc.page_content)
                            st.divider()
            
            with tab2:
                st.header("Explore Retrieved Documents")
                
                # Custom search
                search_query = st.text_input("Search for documents:")
                
                if search_query:
                    results = vectorstore.similarity_search_with_score(search_query, k=6)
                    
                    st.subheader(f"Top {len(results)} Results")
                    for i, (doc, score) in enumerate(results, 1):
                        with st.expander(f"[{i}] {doc.metadata['source_name']} (Score: {score:.4f})"):
                            st.write(doc.page_content)
            
            with tab3:
                st.header("RAG System Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Vectors", vectorstore.index.ntotal)
                
                with col2:
                    st.metric("Vector Dimensions", vectorstore.index.d)
                
                with col3:
                    st.metric("Sources Loaded", len(chunks))
                
                with col4:
                    total_chunks = sum(len(ch) for ch in chunks.values())
                    st.metric("Total Chunks", total_chunks)
                
                st.divider()
                
                st.subheader("📋 Source Breakdown")
                for name, chunk_list in chunks.items():
                    st.write(f"**{name}**: {len(chunk_list)} chunks")
            
            # Example questions in sidebar
            st.sidebar.divider()
            st.sidebar.header("💡 Example Questions")
            example_questions = [
                # General History
                "What is the history of Saudi Arabia?",
                "Who was Ibn Saud and what was his role?",
                
                # First Saudi State (Diriyah)
                "When did the First Saudi State exist?",
                "Why did the First Saudi State fall?",
                "What was Diriyah?",
                
                # Second Saudi State
                "When was the Second Saudi State established?",
                "Who was Turki bin Abdullah?",
                "How long did the Second Saudi State last?",
                
                # Unification
                "How did Saudi Arabia unify?",
                "What happened during the 1902-1932 unification?",
                "What was Ibn Saud's military campaign about?",
                
                # Key Events
                "What role did the Al Saud family play in Saudi history?",
                "What is the connection between Wahhabism and the Saudi state?",
                "When did Riyadh become important in Saudi history?",
                
                # Specific Topics
                "How did geographical factors affect Saudi Arabia's history?",
                "What happened to the Saudi state after 1932?",
            ]
            
            for q in example_questions:
                if st.sidebar.button(q, key=q):
                    st.session_state.selected_question = q
            
            # Process selected question from sidebar
            if "selected_question" in st.session_state:
                question = st.session_state.selected_question
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure your OpenAI API key is valid and you have internet connection")
