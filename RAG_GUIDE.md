# RAG (Retrieval-Augmented Generation) in Streamlit - Complete Guide

## What is RAG?

RAG combines **retrieval** and **generation**:
1. **Retrieval**: Search for relevant documents using vector similarity
2. **Generation**: Use those documents to help an LLM generate accurate answers

## Your Code Analysis

### ✅ What You Have (Saving/Loading):
```python
vectorstore.save_local("saudi_history_faiss")
vectorstore = FAISS.load_local("saudi_history_faiss", embedder, allow_dangerous_deserialization=True)
```
This part is **correct** for persisting FAISS indexes.

### ❌ What You Need to Add:

1. **Embeddings** - Convert text to vectors
2. **Text Splitter** - Break documents into chunks
3. **Vector Store** - Store and retrieve vectors (FAISS)
4. **LLM** - Generate answers (OpenAI, Ollama, HuggingFace, etc.)
5. **Chain** - Connect retriever + LLM

## Complete RAG Pipeline

```
Documents → Split into Chunks → Embed → Store in FAISS
                                          ↓
                                    Query Input
                                          ↓
                                  Similarity Search
                                          ↓
                            Retrieved Documents
                                          ↓
                        Pass to LLM with Prompt
                                          ↓
                                   Final Answer
```

## Installation

```bash
pip install -r requirements_rag.txt
```

## Quick Start Example

### 1. Create Vectorstore (One-time setup)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text("Your document text here...")

# Create and save vectorstore
vectorstore = FAISS.from_texts(chunks, embedder)
vectorstore.save_local("saudi_history_faiss")
```

### 2. Load and Query (Reusable)
```python
# Load existing vectorstore
vectorstore = FAISS.load_local(
    "saudi_history_faiss",
    embedder,
    allow_dangerous_deserialization=True
)

# Retrieve similar documents
docs = vectorstore.similarity_search("What is Saudi history?", k=3)

# Or use as retriever for LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### 3. Generate Answers with LLM
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff", "map_reduce", "refine"
    retriever=retriever
)

answer = qa_chain("Your question here")
```

## Best Practices for Streamlit

### 1. Cache Resources
```python
@st.cache_resource
def load_vectorstore():
    embedder = HuggingFaceEmbeddings()
    return FAISS.load_local("saudi_history_faiss", embedder, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

### 2. Show Retrieved Documents
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents(query)

st.subheader("Retrieved Documents")
for i, doc in enumerate(docs):
    with st.expander(f"Document {i+1}"):
        st.write(doc.page_content)
```

### 3. Stream Responses (for long outputs)
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

## Different LLM Options

### Option 1: Local (Ollama)
```bash
# Install: https://ollama.ai
ollama run llama2
```
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

### Option 2: OpenAI
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key="sk-...", model="gpt-3.5-turbo")
```

### Option 3: Google Generative AI
```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro")
```

### Option 4: HuggingFace
```python
from langchain_community.llms import HuggingFacePipeline
llm = HuggingFacePipeline(model_id="mistralai/Mistral-7B-v0.1")
```

## Chain Types

1. **"stuff"** - Concatenate all docs into prompt (fast, limited context)
2. **"map_reduce"** - Summarize each doc, then combine (handles long docs)
3. **"refine"** - Iteratively refine answers (slower, better for complex Q&A)
4. **"multi_query"** - Generate multiple queries, then combine results

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Change this
    retriever=retriever
)
```

## File Structure
```
streamlit-exercises/
├── rag_example.py          # Full-featured RAG app
├── rag_minimal.py          # Minimal working example
├── requirements_rag.txt    # Dependencies
└── saudi_history_faiss/    # Saved vectorstore (auto-created)
```

## Run the Examples

```bash
# Full example with tabs and multiple features
streamlit run rag_example.py

# Minimal example
streamlit run rag_minimal.py
```

## Troubleshooting

### FAISS not found
```bash
pip install faiss-cpu  # CPU version
# or
pip install faiss-gpu  # GPU version (requires CUDA)
```

### Embeddings model too large
```python
# Use smaller, faster model
HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Ollama connection error
Make sure Ollama is running:
```bash
ollama serve
# or run specific model:
ollama run llama2
```

## Performance Tips

1. **Use smaller embeddings model** for speed (MiniLM instead of BERT)
2. **Increase chunk overlap** for better context preservation
3. **Cache vectorstore** with `@st.cache_resource`
4. **Use streaming** for long outputs
5. **Limit k** (number of retrieved docs) to 3-5 for speed

## Next Steps

1. Start with `rag_minimal.py` to understand basics
2. Expand to `rag_example.py` for full features
3. Add your own documents
4. Experiment with different LLMs and chain types
5. Deploy to Streamlit Cloud

Happy RAG building! 🚀
