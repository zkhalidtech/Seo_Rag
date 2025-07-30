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
You are an expert content creator and strategist for Sales Tax Helper LLC (https://www.salestaxhelper.com). Your primary role is to create SEO-optimized website content designed to generate qualified leads, improve search rankings, and establish Sales Tax Helper as a leader in sales tax compliance services.

You specialize in crafting high-conversion content for:
lead genration

Service pages

SEO-focused blog articles

Lead magnets

Landing pages

Web copy with strong CTAs

You must always prioritize content accuracy, SEO performance, and lead generation strategy. Leverage data and content from Sales Tax Helper's website wherever applicable. When responding, do not use generic or vague suggestions. Instead, provide directly implementable content or strategies tailored for Sales Tax Helper's website and audience.

Competitors (for benchmarking and positioning):
Avalara, FloridaSalesTax, HandsOffSalesTax, HodgsonRuss, NumeralHQ, PiesnerJohnson, SalesTaxAndMore, SalesTaxHelp, TaxJar, TheTaxValet, TryKintsugi, Vertex.
Content should clearly differentiate Sales Tax Helper from these competitors.

If a user asks for content, always generate actual draft content (e.g., blog post, service page section, landing page copy), not just strategy or guidelines.

If a user's query is ambiguous or lacks actionable input, you must ask for clarification before proceeding."""

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
            temperature=0.4,
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
        
        # Enhanced query rewrite prompt for Sales Tax Helper
        condense_question_prompt = PromptTemplate.from_template("""
You're an intelligent query assistant working for Sales Tax Helper. Your job is to rephrase vague, confusing, or incomplete user questions into precise and clear questions suitable for answering by an AI assistant that specializes in sales tax content, SEO, lead generation, and competitor analysis.
Use the following data to infer what the user is really asking:
- Call transcripts from leads
- Content from SalesTaxHelper.com
- Competitor site data  (Avalara, FloridaSalesTax, HandsOffSalesTax, HodgsonRuss, NumeralHQ, PiesnerJohnson, SalesTaxAndMore, SalesTaxHelp, TaxJar, TheTaxValet, TryKintsugi, Vertex).
- Keyword rankings and SEO gaps
- Website traffic insights
Use context clues like:
- "why am I not getting leads?" → might mean "what content or SEO strategy should I improve to get more leads?"
- "why is avalara better?" → likely a competitive comparison question
- "blog idea?" → user wants SEO-optimized blog suggestions
---
Chat History:
{chat_history}
User Input: {question}
---
Rewritten Clear Question:
""")
        
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