"""
Simple PDF Question Answering Agent (No Embeddings Required)
Works directly with PDF text chunks without vector embeddings.
"""
import os
import sys
import glob
from dotenv import load_dotenv
from pdf_processor import load_pdf_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   📄 AI PDF Question Answering Agent                 ║
║   Powered by Google Gemini (No Embeddings)            ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def print_separator():
    """Print a separator line."""
    print("\n" + "─" * 60 + "\n")


def find_pdf_in_folder():
    """Find the first PDF file in the current directory."""
    pdf_files = glob.glob("*.pdf")
    if pdf_files:
        return pdf_files[0]
    return None


def create_chunks(documents, chunk_size=2000, chunk_overlap=200):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def simple_search(query, chunks, top_k=5):
    """Simple keyword-based search through chunks."""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Score chunks based on keyword matches
    scored_chunks = []
    for chunk in chunks:
        content_lower = chunk.page_content.lower()
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score and return top k
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored_chunks[:top_k]]


def ask_question(question, chunks, llm):
    """Ask a question about the PDF content."""
    # Find relevant chunks
    relevant_chunks = simple_search(question, chunks, top_k=4)
    
    if not relevant_chunks:
        return {
            "answer": "I couldn't find relevant information in the PDF to answer your question.",
            "sources": []
        }
    
    # Prepare context from relevant chunks
    context = "\n\n".join([
        f"[Page {chunk.metadata.get('page', 'Unknown')}]\n{chunk.page_content}"
        for chunk in relevant_chunks
    ])
    
    # Create prompt
    prompt = f"""You are a helpful AI assistant analyzing a PDF document. 
Based on the following excerpts from the document, please answer the question.
If the answer is not in the provided context, say so honestly.
Always mention the page number when referencing information.

Context from PDF:
{context}

Question: {question}

Answer:"""
    
    # Get response from LLM
    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "page": chunk.metadata.get("page", "Unknown"),
                    "content": chunk.page_content[:200] + "..."
                }
                for chunk in relevant_chunks
            ]
        }
    except Exception as e:
        return {
            "answer": f"Error getting response: {str(e)}",
            "sources": []
        }


def display_response(response):
    """Display the agent's response."""
    print("\n🤖 Answer:")
    print_separator()
    print(response["answer"])
    
    # Display sources
    if response["sources"]:
        print_separator()
        print("📚 Sources:")
        for idx, source in enumerate(response["sources"], 1):
            print(f"\n  [{idx}] Page {source['page']}:")
            print(f"      {source['content']}")
    
    print_separator()


def interactive_mode(chunks, llm):
    """Run interactive question-answering mode."""
    print("\n✨ Ready to answer your questions!")
    print("   Type 'quit' or 'exit' to end the session")
    print_separator()
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Your Question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the PDF QA Agent. Goodbye!\n")
                break
            
            # Skip empty questions
            if not question:
                continue
            
            # Get answer
            print("\n🔄 Processing your question...")
            response = ask_question(question, chunks, llm)
            
            # Display response
            display_response(response)
        
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using the PDF QA Agent. Goodbye!\n")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def main():
    """Main function."""
    print_banner()
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("   Please create a .env file with your Google API key")
        sys.exit(1)
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = find_pdf_in_folder()
        if pdf_path:
            print(f"📄 Found PDF: {pdf_path}")
        else:
            print("❌ Error: No PDF file found in the current directory")
            print("   Please place a PDF file in this folder or specify the path:")
            print("   python simple_qa_agent.py <path_to_pdf>")
            sys.exit(1)
    
    # Validate PDF path
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.lower().endswith('.pdf'):
        print("❌ Error: File must be a PDF")
        sys.exit(1)
    
    try:
        # Load PDF
        print(f"\n🔄 Loading PDF: {pdf_path}")
        print_separator()
        documents = load_pdf_documents(pdf_path)
        print(f"✅ Successfully loaded {len(documents)} pages\n")
        
        # Create chunks
        print("🔄 Splitting document into chunks...")
        chunks = create_chunks(documents)
        print(f"✅ Created {len(chunks)} chunks\n")
        
        # Initialize LLM
        print("🔄 Initializing Gemini...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0
        )
        print("✅ Ready!\n")
        
        print_separator()
        
        # Run interactive mode
        interactive_mode(chunks, llm)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
