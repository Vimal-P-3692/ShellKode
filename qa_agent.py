"""
QA Agent Module
Implements the LangChain agent for question answering with conversational memory.
"""
from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


class PDFQAAgent:
    """AI agent for answering questions about PDF documents."""
    
    def __init__(
        self,
        retriever,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the QA agent.
        
        Args:
            retriever: LangChain retriever object
            model_name: Name of the Google Gemini model
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
        """
        self.retriever = retriever
        
        # Initialize the language model
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create custom prompt template
        self.qa_prompt = PromptTemplate(
            template="""You are an AI assistant helping to answer questions about a PDF document.
Use the following pieces of context from the document to answer the question at the end.
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
Always provide the page number(s) from the metadata when referencing information from the document.

Context:
{context}

Question: {question}

Helpful Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create the conversational retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the PDF document.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        try:
            result = self.qa_chain({"question": question})
            
            # Format the response
            response = {
                "question": question,
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "sources": self._format_sources(result.get("source_documents", []))
            }
            
            return response
        
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "source_documents": [],
                "sources": []
            }
    
    def _format_sources(self, source_docs: List) -> List[Dict[str, Any]]:
        """
        Format source documents for display.
        
        Args:
            source_docs: List of source documents
            
        Returns:
            List of formatted source information
        """
        sources = []
        for doc in source_docs:
            source_info = {
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            }
            sources.append(source_info)
        
        return sources
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
        print("Conversation memory cleared")
    
    def get_chat_history(self) -> List:
        """
        Get the chat history.
        
        Returns:
            List of chat messages
        """
        return self.memory.chat_memory.messages


class SimpleQAAgent:
    """Simplified QA agent without conversational memory."""
    
    def __init__(
        self,
        retriever,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0
    ):
        """
        Initialize the simple QA agent.
        
        Args:
            retriever: LangChain retriever object
            model_name: Name of the Google Gemini model
            temperature: Temperature for response generation
        """
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    
    def ask(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Ask a question about the PDF document.
        
        Args:
            question: The question to ask
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing answer and source documents
        """
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""Based on the following context from a PDF document, please answer the question.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        # Get response from LLM
        response = self.llm.predict(prompt)
        
        return {
            "question": question,
            "answer": response,
            "source_documents": docs,
            "sources": [{"page": doc.metadata.get("page"), "source": doc.metadata.get("source")} for doc in docs]
        }
