# PDFChatter (Conversational RAG for PDFs 📚)

Ever wished you could chat with a PDF like you do with a real expert? This Retrieval-Augmented Generation (RAG) system makes that possible! It **understands your queries, retrieves the most relevant sections, and generates insightful responses**—all in an interactive and user-friendly interface.

## 🚀 How It Works
✅ **Multi-query retrieval** ensures broad and relevant search results  
✅ **Reranking with a sentence-transformer model** guarantees top-quality context  
✅ **Mistral 7B** generates clear, informative responses 

## 🔧 Tech Stack

- **LangChain** for seamless integration of retrieval and generation components
- **LangSmith** for tracing and performance monitoring
- **ChromaDB** as the vector database for efficient document retrieval
- **Mistral 7B** for generating high-quality responses
- **Streamlit** for an intuitive user interface


## 🛠️ **Installation and Setup**

1. Clone the respository:
```
https://github.com/minkvirparia/PDFChatter.git
```

2. Create and Activate a Virtual Environment

```
python -m venv env
source env/bin/activate  # For Linux/Mac
env\Scripts\activate     # For Windows

```


3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Create .env file and add langsmith and huggingface token in following manner:

```
LANGCHAIN_TRACING_V2= true
LANGCHAIN_ENDPOINT = https://api.smith.langchain.com
LANGCHAIN_API_KEY = your_langchain_key
LANGCHAIN_PROJECT = pdf-chatter
HF_API_KEY = your_huggingface_key
```

5. Run the app:

```
streamlit run app.py
```


## 🎯 Features
- 🔍 **Ask any question** related to your PDF
- 📄 **Handles large documents** efficiently
- 🤖 **AI-powered conversational interface**
- 📈 **Optimized retrieval and ranking** for relevant answers

## 📝 Usage
1. Upload a PDF document.
2. Ask any question related to the document.
3. Get accurate, AI-generated responses in seconds.
