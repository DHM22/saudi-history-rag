# 🏛️ Saudi Arabia History RAG - Streamlit App

A Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain that answers questions about Saudi Arabia's history using Wikipedia sources.

## Features

- **🤖 Ask Questions Tab**: Query the knowledge base with natural language questions
- **🔍 Explore Documents Tab**: Search and view retrieved documents with similarity scores
- **📊 Stats Tab**: View system statistics (total vectors, dimensions, chunk breakdown)
- **💡 Example Questions**: Pre-configured contextual questions organized by historical period
- **🔐 API Configuration**: Secure OpenAI API key input in sidebar
- **⚙️ Tunable Retrieval**: Adjust the number of documents (k) retrieved per query

## Data Sources

The app loads information from 5 Wikipedia articles:

1. [History of Saudi Arabia](https://en.wikipedia.org/wiki/History_of_Saudi_Arabia)
2. [First Saudi State](https://en.wikipedia.org/wiki/First_Saudi_state)
3. [Second Saudi State](https://en.wikipedia.org/wiki/Second_Saudi_state)
4. [Unification of Saudi Arabia (1902-1932)](https://en.wikipedia.org/wiki/Unification_of_Saudi_Arabia)
5. [Ibn Saud (Founder)](https://en.wikipedia.org/wiki/Ibn_Saud)

## Architecture

```
Wikipedia Sources
       ↓
   WebBaseLoader (BeautifulSoup4)
       ↓
RecursiveCharacterTextSplitter (800 chunk size, 120 overlap)
       ↓
OpenAI text-embedding-3-small
       ↓
FAISS VectorStore (cosine similarity)
       ↓
LangChain Retriever (top-k documents)
       ↓
ChatGPT-4 Turbo (context + question)
       ↓
    Answer + Sources
```

## Installation

### Prerequisites
- Python 3.9+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/streamlit-exercises.git
cd streamlit-exercises
```

2. Create a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate    # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements_rag.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run rag_saudi_history.py
```

2. Open your browser to `http://localhost:8501`

3. Enter your OpenAI API key in the sidebar

4. Ask questions or click example questions from the sidebar

## Configuration

### Sidebar Options
- **OpenAI API Key**: Your private API key (required)
- **Number of documents (k)**: How many context chunks to retrieve (1-10, default 6)
- **Clear Cache**: Reset cached documents and embeddings

### Example Questions

Pre-configured questions cover:
- **General History**: Saudi Arabia overview, Ibn Saud
- **First Saudi State**: Diriyah, Turki bin Abdullah
- **Second Saudi State**: Establishment, duration
- **Unification (1902-1932)**: Military campaigns, territorial expansion
- **Key Events**: Al Saud family, Wahhabism, Riyadh
- **Specific Topics**: Geography, post-1932 history

## Files

| File | Purpose |
|------|---------|
| `rag_saudi_history.py` | **Main app** - Production RAG for Saudi history |
| `requirements_rag.txt` | Python dependencies |
| `RAG_GUIDE.md` | Comprehensive RAG concepts documentation |
| `rag_minimal.py` | Minimal RAG example (educational) |
| `rag_example.py` | Full-featured RAG with tabs (educational) |
| `app.py` | Basic Streamlit demo |
| `RAG_History_of_Saudi_Arabia.ipynb` | Original Jupyter notebook reference |

## Technical Stack

- **Streamlit**: Web UI framework
- **LangChain**: RAG orchestration and LLM integration
- **FAISS**: Vector similarity search
- **OpenAI**: Embeddings (text-embedding-3-small) and LLM (gpt-4-turbo)
- **BeautifulSoup4**: HTML parsing from Wikipedia
- **Langchain-text-splitters**: Document chunking

## How RAG Works

1. **Indexing Phase** (first run):
   - Wikipedia articles are loaded and parsed
   - Documents are split into 800-character chunks with 120-character overlap
   - Each chunk is embedded using OpenAI's text-embedding-3-small
   - Embeddings are stored in FAISS for fast retrieval

2. **Query Phase** (user asks question):
   - User question is embedded using same embedder
   - FAISS finds k most similar document chunks (cosine similarity)
   - Retrieved chunks are formatted as context
   - Context + question sent to GPT-4 Turbo
   - LLM generates answer constrained to only use provided context

3. **Caching** (performance):
   - Streamlit caches documents, embeddings, vectorstore, and QA chain
   - Second and subsequent runs are much faster
   - Use "Clear Cache" button to refresh

## Troubleshooting

### "Please provide OpenAI API key"
- Enter your valid OpenAI API key in the sidebar
- Ensure you have remaining API credits

### "Cannot hash argument 'vectorstore'"
- This is handled by using underscore prefix on vectorstore parameter
- If you see this error, clear cache and restart Streamlit

### "No retrieved documents"
- Increase k value in sidebar (retrieve more chunks)
- Try simpler keywords in your question
- Check internet connection (needed to load Wikipedia)

### Slow first run
- First run takes 30-60 seconds as it embeds all documents
- Subsequent runs use cache and are much faster
- Clear cache to force re-embedding

## Future Enhancements

- [ ] Streaming responses for better UX
- [ ] Custom document upload
- [ ] Export results to PDF
- [ ] Different language models
- [ ] Citation tracking
- [ ] Question suggestions

## License

MIT

## Author

Created for Streamlit exercises with LangChain RAG patterns

## Support

For issues, questions, or improvements, please open an issue on GitHub.
