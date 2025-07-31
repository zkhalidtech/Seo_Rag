import streamlit as st
import os
import pandas as pd
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
import pickle
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced System prompt for actionable SEO AI Assistant
SYSTEM_PROMPT = """You are an expert SEO Content Strategist and Implementation Specialist for Sales Tax Helper LLC. Your role is to analyze, create, and provide SPECIFIC, ACTIONABLE content and improvements.

CRITICAL INSTRUCTIONS:
1. ALWAYS provide actual content, not suggestions
2. When asked about improvements, give EXACT text to replace current content
3. Include specific HTML/CSS when relevant
4. Reference exact page URLs and sections
5. Write in first person as if you ARE the business owner

Your knowledge base includes:
- Sales Tax Helper website content and structure
- Competitor analysis from: Avalara, TaxJar, Vertex, etc.
- SEMRush keyword data and rankings
- Google Analytics user behavior data
- Call transcripts from actual leads

RESPONSE FORMAT:
- For content requests: Write the FULL content piece
- For improvements: Provide BEFORE/AFTER comparisons
- For technical issues: Give exact code snippets
- For strategy: Include specific KPIs and timelines

EXAMPLE RESPONSES:

Bad Response: "You should improve your homepage headline"
Good Response: "Replace your current headline 'Welcome to Sales Tax Helper' with: 'Save $50,000+ Annually on Sales Tax Compliance - Get Your Free Tax Nexus Analysis in 24 Hours'"

Bad Response: "Add more CTAs to your service page"
Good Response: "Add this CTA after paragraph 3 on /services/nexus-analysis:
<div class='cta-box'>
  <h3>Stop Overpaying on Sales Tax</h3>
  <p>Our clients save an average of $4,200/month. See your potential savings:</p>
  <button onclick='openCalendly()'>Get Free Tax Analysis →</button>
</div>"

When analyzing issues:
- Reference specific competitor advantages with data
- Cite exact keyword gaps and search volumes
- Quote actual customer pain points from call transcripts
- Provide conversion rate benchmarks

Remember: You're not an advisor - you're the implementation expert who writes the actual content and code."""

# Additional prompt for content analysis
CONTENT_ANALYSIS_PROMPT = """Based on the retrieved context, analyze the following aspects:
1. Current content weaknesses compared to top 3 competitors
2. Missing keywords with search volume > 1000/month
3. Conversion optimization opportunities
4. Technical SEO issues

Provide specific fixes with exact implementation details."""

def get_enhanced_retriever(vectorstore):
    """Create an enhanced retriever with better search capabilities"""
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 15,
            "score_threshold": 0.3,
            "fetch_k": 30
        }
    )

def format_context_with_metadata(docs: List[Document]) -> str:
    """Format retrieved documents with their metadata for better context"""
    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata_str = json.dumps(doc.metadata, indent=2)
        formatted_docs.append(f"""
Document {i+1}:
Source: {doc.metadata.get('file', 'Unknown')}
Content: {doc.page_content}
Metadata: {metadata_str}
---""")
    return "\n".join(formatted_docs)

@st.cache_resource
def initialize_rag_with_memory():
    """Initialize the RAG system with enhanced memory and retrieval capabilities"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found in secrets. Please add OPENAI_API_KEY to your Streamlit secrets.")
            return None, None
        
        # Initialize LLM with higher temperature for more creative content
        llm = ChatOpenAI(
            model="gpt-4.1",  # Using GPT-4 for better quality
            temperature=0.7,
            api_key=api_key,
            max_tokens=2000  # Allow longer responses
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
        
        # Load FAISS index
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
        
        # Create enhanced retriever
        retriever = get_enhanced_retriever(vectorstore)
        
        # Initialize memory with higher token limit
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Enhanced query rewrite prompt
        condense_question_prompt = PromptTemplate.from_template("""
You are analyzing a query for Sales Tax Helper's AI system. Your job is to expand vague queries into specific, actionable requests.

Context available in the system:
- Sales Tax Helper website pages and content
- Competitor content and strategies
- SEMRush data: keyword rankings, gaps, search volumes
- Google Analytics: user behavior, conversion rates
- Call transcripts: customer pain points, objections

Query transformation rules:
- "improve my homepage" → "analyze my homepage content against top 3 competitors and rewrite the hero section with higher-converting copy"
- "why no leads?" → "identify specific content gaps causing low conversions and provide replacement content"
- "blog ideas" → "generate 5 high-traffic blog titles based on keyword gaps and write the introduction for the top one"
- "fix my CTA" → "analyze current CTAs, benchmark against competitors, and provide 3 new CTA variations with A/B test plan"

Chat History:
{chat_history}

Original Query: {question}

Rewritten Specific Query:""")
        
        # Enhanced QA prompt with examples
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Context from knowledge base:
{context}

Chat History: {chat_history}

User Query: {question}

Instructions:
1. Use the context to provide SPECIFIC, IMPLEMENTABLE solutions
2. Reference exact data points from the context
3. Write actual content, not just suggestions
4. Include metrics and timelines where relevant
5. Format response for immediate implementation

Your response:""")
        ])
        
        # Create conversational retrieval chain with custom formatting
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,  # Return sources for transparency
            verbose=False
        )
        
        logger.info("Enhanced RAG system initialized successfully!")
        return qa_chain, memory
    
    except Exception as e:
        st.error("Failed to initialize the AI assistant. Please contact support.")
        logger.error(f"RAG initialization error: {str(e)}", exc_info=True)
        return None, None

def display_sources(source_documents):
    """Display source documents in an expandable section"""
    if source_documents:
        with st.expander("📚 Sources Used", expanded=False):
            for i, doc in enumerate(source_documents[:5]):  # Show top 5 sources
                st.markdown(f"**Source {i+1}:** {doc.metadata.get('file', 'Unknown')}")
                st.text(doc.page_content[:200] + "...")

def main():
    st.set_page_config(
        page_title="Sales Tax Helper AI - Content & SEO Specialist",
        page_icon="🚀",
        layout="wide"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
    }
    .example-query {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    .example-query:hover {
        background-color: #e0e2e6;
        transform: translateX(5px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Sales Tax Helper AI Assistant</h1>
        <p>Get instant, actionable content and SEO improvements based on real data</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Sidebar with example queries
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Data Sources")
        st.markdown("""
        - ✅ Sales Tax Helper website
        - ✅ Competitor analysis
        - ✅ SEMRush keyword data
        - ✅ Customer call transcripts
        """)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    display_sources(message["sources"])
    
    # Handle example query selection
    if "next_query" in st.session_state:
        query_to_process = st.session_state.next_query
        del st.session_state.next_query
    else:
        query_to_process = st.chat_input("Ask for specific content, improvements, or analysis...")
    
    # React to user input
    if query_to_process:
        # Display user message
        with col1:
            st.chat_message("user").markdown(query_to_process)
        st.session_state.messages.append({"role": "user", "content": query_to_process})
        
        # Generate response
        with col1:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data and generating content..."):
                    try:
                        # Use the chain with memory
                        result = qa_chain.invoke({"question": query_to_process})
                        response = result['answer']

                        
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                        })
                        
                    except Exception as e:
                        error_message = "I encountered an error. Please try rephrasing your question or contact support."
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        logger.error(f"Query processing error: {str(e)}", exc_info=True)
    


if __name__ == "__main__":
    main()