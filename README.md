# AI PDF Question Answering Agent

An intelligent PDF Question Answering system built with LangChain and **Google Gemini** that allows you to ask questions about your PDF documents.

## Features

- 📄 Load and process PDF documents
- 🤖 AI-powered question answering using Google Gemini via LangChain agents
- 🔍 Semantic search with FAISS vector store
- 💬 Conversational memory for context-aware responses
- 🖥️ Simple CLI interface

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

## Usage

### Option 1: Auto-detect PDF
Place a PDF file in the current folder and run:
```bash
python pdf_qa_agent.py
```

### Option 2: Specify PDF path
```bash
python pdf_qa_agent.py path/to/your/document.pdf
```

The agent will:
- Load and process your PDF
- Create embeddings and vector store
- Start an interactive Q&A session

Type your questions and get answers with source references!

## Commands

- Type your question to get an answer
- Type `clear` to clear conversation memory
- Type `quit` or `exit` to end the session

## Project Structure

- `pdf_processor.py` - PDF loading and text extraction
- `vector_store.py` - Document chunking and vector store management
- `qa_agent.py` - LangChain agent implementation
- `pdf_qa_agent.py` - CLI interface

## How It Works

1. Upload a PDF document
2. The system splits the document into chunks
3. Creates embeddings using Google's embedding model and stores them in a FAISS vector store
4. When you ask a question, the agent retrieves relevant chunks
5. Google Gemini processes the context and generates an answer

## Requirements

- Python 3.8+
- Google API key (Gemini)
