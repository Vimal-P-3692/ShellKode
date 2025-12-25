# AI PDF Question Answering Agent

A powerful PDF Question Answering system built **without LangChain agents**, using OpenAI's API, vector embeddings, and similarity search. This implementation gives you full control over the QA pipeline.

## 🌟 Features

- **PDF Processing**: Extract text from PDF documents
- **Smart Chunking**: Split documents into overlapping chunks for better context
- **Vector Embeddings**: Create embeddings using OpenAI's embedding models
- **Semantic Search**: Find relevant content using cosine similarity
- **Question Answering**: Get accurate answers based on PDF content
- **Save/Load Embeddings**: Process once, query multiple times
- **Interactive Mode**: Chat-style interface for continuous Q&A
- **No LangChain Required**: Custom implementation with full transparency

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- PDF document(s) to analyze

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd Task2

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

```bash
# Option 1: Environment variable
export OPENAI_API_KEY='your-api-key-here'

# Option 2: Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. Basic Usage

```python
from pdf_qa_agent import PDFQuestionAnsweringAgent

# Initialize the agent
agent = PDFQuestionAnsweringAgent()

# Process a PDF
agent.process_pdf("your_document.pdf")

# Ask questions
result = agent.answer_question("What is the main topic?")
print(result['answer'])
```

## 📖 Detailed Usage

### Process a PDF Document

```python
from pdf_qa_agent import PDFQuestionAnsweringAgent

agent = PDFQuestionAnsweringAgent(
    model="gpt-4",  # or "gpt-3.5-turbo" for faster responses
    embedding_model="text-embedding-ada-002"
)

# Process PDF with custom parameters
agent.process_pdf(
    pdf_path="document.pdf",
    chunk_size=1000,  # Characters per chunk
    overlap=200       # Overlap between chunks
)
```

### Ask Questions

```python
# Simple question
result = agent.answer_question("What are the key findings?")
print(result['answer'])

# With custom parameters
result = agent.answer_question(
    question="Explain the methodology",
    top_k=5,          # Number of relevant chunks to consider
    temperature=0.3   # Lower = more focused, Higher = more creative
)

print(f"Answer: {result['answer']}")
print(f"Source: {result['source']}")
```

### View Relevant Context

```python
result = agent.answer_question("What is the conclusion?")

# Display the answer
print(f"Answer: {result['answer']}\n")

# Show relevant chunks used for answering
for i, (chunk, score) in enumerate(result['relevant_chunks'], 1):
    print(f"[Chunk {i}] Similarity: {score:.3f}")
    print(chunk[:200] + "...\n")
```

### Save and Load Embeddings

```python
# Process once and save
agent.process_pdf("large_document.pdf")
agent.save_embeddings("embeddings.pkl")

# Later, load without reprocessing
agent = PDFQuestionAnsweringAgent()
agent.load_embeddings("embeddings.pkl")
result = agent.answer_question("Your question here")
```

### Interactive Mode

```python
agent = PDFQuestionAnsweringAgent()
agent.process_pdf("document.pdf")

# Start interactive Q&A session
agent.interactive_mode()
```

## 🔧 Configuration Options

### Model Selection

```python
agent = PDFQuestionAnsweringAgent(
    model="gpt-4",              # Best quality
    # model="gpt-3.5-turbo",    # Faster, cheaper
    embedding_model="text-embedding-ada-002"
)
```

### Chunking Strategy

```python
agent.process_pdf(
    pdf_path="doc.pdf",
    chunk_size=1500,   # Larger chunks = more context per chunk
    overlap=300        # More overlap = better context continuity
)
```

### Answer Generation

```python
result = agent.answer_question(
    question="Your question",
    top_k=5,           # More chunks = more context (but slower)
    temperature=0.3    # 0 = deterministic, 1 = creative
)
```

## 📁 Project Structure

```
Task2/
├── pdf_qa_agent.py       # Main agent implementation
├── example_usage.py      # Usage examples
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎯 How It Works

1. **PDF Loading**: Extract text from PDF using PyPDF2
2. **Text Chunking**: Split text into overlapping chunks for better context
3. **Embedding Creation**: Convert chunks to vector embeddings using OpenAI
4. **Question Processing**: Convert user question to embedding
5. **Similarity Search**: Find most relevant chunks using cosine similarity
6. **Answer Generation**: Use GPT to generate answer based on relevant context

## 💡 Example Use Cases

### Academic Research
```python
agent.process_pdf("research_paper.pdf")
agent.answer_question("What methodology was used?")
agent.answer_question("What are the limitations?")
agent.answer_question("What are the future research directions?")
```

### Legal Documents
```python
agent.process_pdf("contract.pdf")
agent.answer_question("What are the payment terms?")
agent.answer_question("What is the termination clause?")
```

### Technical Documentation
```python
agent.process_pdf("manual.pdf")
agent.answer_question("How do I install the software?")
agent.answer_question("What are the system requirements?")
```

### Business Reports
```python
agent.process_pdf("annual_report.pdf")
agent.answer_question("What was the revenue growth?")
agent.answer_question("What are the main risks?")
```

## 🔍 Advanced Features

### Batch Processing Multiple PDFs

```python
agent = PDFQuestionAnsweringAgent()

pdfs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
all_results = []

for pdf in pdfs:
    agent.process_pdf(pdf)
    result = agent.answer_question("What is the main topic?")
    all_results.append({
        'pdf': pdf,
        'answer': result['answer']
    })

for r in all_results:
    print(f"{r['pdf']}: {r['answer']}")
```

### Custom Prompt Engineering

Modify the `answer_question` method in `pdf_qa_agent.py` to customize the prompt:

```python
prompt = f"""You are an expert analyst. Based on the following context,
provide a detailed answer with specific examples.

Context:
{context}

Question: {question}

Detailed Answer:"""
```

## ⚙️ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |

## 🐛 Troubleshooting

### "OpenAI API key must be provided"
Set your API key: `export OPENAI_API_KEY='your-key'`

### "PDF file not found"
Check the file path is correct and the file exists

### "No PDF has been processed"
Call `agent.process_pdf()` before asking questions

### Out of Memory
Reduce `chunk_size` or process smaller PDFs

## 📊 Performance Tips

1. **Use GPT-3.5-turbo** for faster/cheaper responses (vs GPT-4)
2. **Save embeddings** to avoid reprocessing same PDFs
3. **Adjust chunk_size** based on document type:
   - Dense technical docs: 800-1000 chars
   - General text: 1000-1500 chars
4. **Optimize top_k**: Start with 3-5, increase if answers lack context
5. **Lower temperature** (0.1-0.3) for factual accuracy

## 🆚 Why Not LangChain?

This implementation gives you:
- ✅ Full control over the entire pipeline
- ✅ No hidden abstractions
- ✅ Easy to debug and customize
- ✅ Minimal dependencies
- ✅ Transparent data flow
- ✅ Better understanding of how it works

## 📝 API Cost Estimation

| Operation | Model | Approx Cost |
|-----------|-------|-------------|
| Embeddings | text-embedding-ada-002 | $0.0001/1K tokens |
| Answers | GPT-4 | $0.03/1K tokens (input) |
| Answers | GPT-3.5-turbo | $0.0015/1K tokens (input) |

**Example**: A 50-page PDF (~25K words):
- Embeddings: ~$0.10
- 10 questions (GPT-4): ~$0.30-0.50
- Total: ~$0.40-0.60

## 🤝 Contributing

Feel free to:
- Add support for other document formats (DOCX, TXT, etc.)
- Implement different embedding models
- Add caching mechanisms
- Improve chunking strategies
- Add support for multi-modal documents (images, tables)

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- OpenAI for GPT and embedding models
- PyPDF2 for PDF processing

## 📮 Support

For issues or questions:
1. Check the troubleshooting section
2. Review example_usage.py for working examples
3. Ensure all dependencies are installed correctly

---

**Happy Question Answering! 🎉**
