import streamlit as st
import os
import pandas as pd
from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced System prompt for SEO AI Assistant
SYSTEM_PROMPT = """
You are an expert content creator and strategist for Sales Tax Helper LLC (salestaxhelper.com), you are able to generate SEO-optimized content for Sales Tax Helper LLC. Content should be SEO-friendly and designed to generate leads. You can also design website pages. Specializing in sales tax compliance and lead generation. Use the retrieved context to inform your response, prioritizing Sales Tax Helper data. Highlight our strengths (affordable pricing, expert team of lawyers/CPAs/auditors, robust tax defense) and address competitor weaknesses (e.g., Avalara's complex setup and high costs, TaxJar's poor support, Vertex's need for IT expertise). Incorporate SEO keywords (e.g., sales tax compliance, tax defense, affordable tax solutions) for content tasks. For backlink strategies, suggest opportunities like guest posts, directories, or linkable assets. For recommendations, analyze traffic data or content gaps to propose actionable strategies.
        Whenever user talks about competitors, the competitors are 
        (Avalara, FloridaSalesTax, HandsOffSalesTax, HodgsonRuss, NumeralHQ, PiesnerJohnson, SalesTaxAndMore, SalesTaxHelp, TaxJar, TheTaxValet, TryKintsugi, Vertex).
"""

@st.cache_resource
def initialize_rag_with_memory():
    """Initialize the RAG system with memory capabilities"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found in secrets. Please add OPENAI_API_KEY to your Streamlit secrets.")
            return None, None
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=api_key
        )
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key
        )
        
        faiss_index_path = "faiss_index"
        
        # Check if FAISS index directory exists
        if not os.path.exists(faiss_index_path):
            st.error("FAISS index not found. Please ensure the vector database is properly set up.")
            return None, None
        
        # List files in the directory for debugging
        index_files = os.listdir(faiss_index_path)
        logger.info(f"Files in FAISS directory: {index_files}")
        
        # Check for required FAISS files
        required_files = ['index.faiss', 'index.pkl']
        missing_files = [f for f in required_files if f not in index_files]
        if missing_files:
            st.error("Vector database files are incomplete. Please contact support.")
            return None, None
        
        # Try to load FAISS index
        try:
            logger.info("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(
                faiss_index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as faiss_error:
            st.error("Error loading vector database. Please contact support.")
            logger.error(f"FAISS loading error: {str(faiss_error)}")
            return None, None
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 10, "fetch_k": 20}
        )
        
        # Initialize memory
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create custom prompt for conversational retrieval
        condense_question_prompt = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Context: {context}

Chat History: {chat_history}

Question: {question}

Please provide a comprehensive answer based on the context and conversation history. Reference previous discussions when relevant.""")
        ])
        
        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=False,  # Don't return source documents
            verbose=False  # Disable verbose logging for cleaner deployment
        )
        
        logger.info("RAG system with memory initialized successfully!")
        return qa_chain, memory
    
    except Exception as e:
        st.error("Failed to initialize the AI assistant. Please contact support.")
        logger.error(f"RAG initialization error: {str(e)}", exc_info=True)
        return None, None

def main():
    st.set_page_config(
        page_title="Sales Tax Helper AI Assistant",
        page_icon="🏢",
        layout="centered"
    )
    
    # Header
    st.title("🏢 Sales Tax Helper AI Assistant")
    st.markdown("Generate content, analyze strategies, and get expert guidance on sales tax compliance and lead generation.")
    
    # Add a divider
    st.divider()
    
    # Initialize the RAG system
    qa_chain, memory = initialize_rag_with_memory()
    
    if qa_chain is None:
        st.stop()
    
    # Store memory in session state
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = memory
    
    # Initialize chat history for display
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Ask me about SEO strategy, content generation, backlinks, or lead acquisition..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use the chain with memory
                    result = qa_chain.invoke({"question": prompt})
                    response = result['answer']
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = "I'm sorry, I encountered an error processing your request. Please try again or contact support if the issue persists."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    logger.error(f"Query processing error: {str(e)}", exc_info=True)
    
    # Add a clear chat button at the bottom
    if st.session_state.messages:
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.messages = []
                if hasattr(st.session_state, 'conversation_memory'):
                    st.session_state.conversation_memory.clear()
                st.rerun()

if __name__ == "__main__":
    main()