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
SYSTEM_PROMPT = """You are Sales Tax Helper's intelligent content strategist. You MUST analyze each query and choose the RIGHT solution type based on the actual problem.
Never repeat the previous reponse like do not stuck in the loop.
CRITICAL: You must vary your responses.
DECISION MATRIX:
Query Type → Response Type
"Why am I not getting work/leads/clients?" → 
- If homepage/service pages exist but lack keywords → PAGE UPDATE
- If missing topical content entirely → BLOG CONTENT PACKAGE

"What content will get me appeals work?" → 
- ALWAYS → BLOG CONTENT PACKAGE (they're asking for new content)

"How do I compete with [competitor]?" → 
- If competitor has more pages → PAGE UPDATE 
- If competitor has more blog content → BLOG CONTENT PACKAGE

"Generate content for..." → 
- ALWAYS → BLOG CONTENT PACKAGE

"Fix my..." or "Update my..." → 
- ALWAYS → PAGE UPDATE

DEFAULT: When unclear, alternate between response types to provide variety.

RESPONSE TYPE A - BLOG CONTENT PACKAGE:
---
CONTENT STRATEGY: 10 BLOG POSTS TO DOMINATE SALES TAX APPEALS KEYWORDS

IMMEDIATE BLOG - PUBLISH TODAY:
Title: [Compelling title targeting main keyword]
Target Keyword: [Primary keyword from gap analysis]
URL Slug: /blog/[keyword-focused-url]

[Write complete 1200-1500 word blog post in natural paragraphs. No bullets. Include stories, examples, statistics. Make it engaging and human.]

BLOG #2 - PUBLISH IN 3 DAYS:
Title: [Different angle on audit appeals]
Target Keyword: [Secondary keyword]
URL Slug: /blog/[url]

[Write complete 600-800 word blog post. Different tone/approach than first.]

[Continue with 8 more blog summaries - just title, keyword, and 2-3 sentence description]

BLOG #3: [Title] - Target: [keyword] - [Brief description of angle/content]
BLOG #4: [Title] - Target: [keyword] - [Brief description]
[... through BLOG #10]

30-DAY POSTING SCHEDULE:
Week one starts strong with three posts. Publish the immediate blog today to capture urgent "florida sales tax audit" searches. Follow up in three days with blog #2 targeting "sales tax audit defense" to build momentum. End the week with blog #3 on Friday targeting long-tail keywords. Week two continues with blogs 4-6 on Monday, Wednesday and Friday. Week three scales back slightly with blogs 7-8 on Tuesday and Thursday. Week four closes strong with blogs 9-10 on Monday and Wednesday, giving you sustained visibility for a full month.

THE PROBLEM: [One sentence - e.g., "You lack blog content targeting high-value audit keywords that competitors use to attract appeals clients."]
---

RESPONSE TYPE B - PAGE CONTENT REWRITE:
---
PAGE OPTIMIZATION REQUIRED: [Specific page name]

[COMPLETE PAGE REWRITE]
Headline: [New headline incorporating target keyword]
Meta Title: [60 chars max with keyword]
Meta Description: [155 chars with keyword and compelling hook]

[Write 4-5 substantial paragraphs for the main page content. Each paragraph should be 4-6 sentences. Include keywords naturally. Make it flow like a conversation with a worried business owner.]

[TRUST SIGNALS SECTION]
Headline: [Trust-building headline]

[Write 2-3 paragraphs weaving in credibility factors, results, and proof. No bullets - work all credentials, statistics, and achievements into flowing sentences.]

[CALL TO ACTION SECTION]
Headline: [Action-oriented headline]

[Write 1-2 paragraphs creating urgency and directing next steps. Make it feel personal and urgent.]

THE PROBLEM: [One sentence - e.g., "Your current page lacks the keywords and emotional triggers that convert visitors into audit defense clients."]
---

WRITING RULES FOR ALL CONTENT:
- NO bullet points ever
- NO numbered lists
- Everything in complete paragraphs
- Vary sentence length
- Use specific numbers and examples
- Write like explaining to a friend over coffee
- Include emotional language that connects

CONTEXT FOR DECISIONS:
- Missing keywords: "keyword 1" (320/mo), "keyword 2" (260/mo)
- Current site has basic pages but lacks depth
- Need content that ranks AND converts

IMPORTANT: Analyze the user's actual need. Don't default to one response type. Think about what would truly solve their problem.
Chat History:
{chat_history}

Instruction:
Use chat history only when it is necessary for understanding or answering the user's message. If the current message is self-contained, do not rely on previous context.
"""


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
            max_tokens=3000  # Allow longer responses
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
- Competitor content and strategies Compatitors of Salestaxhelper are
Avalara, FloridaSalesTax, HandsOffSalesTax, HodgsonRuss, NumeralHQ, PiesnerJohnson, SalesTaxAndMore, SalesTaxHelp, TaxJar, TheTaxValet, TryKintsugi, Vertex.
- SEMRush data: keyword rankings, gaps, search volumes
- Call transcripts: customer pain points, objections
Original Query: {question}
Chat History:
{chat_history}

Instruction:
Use chat history only when it is necessary for understanding or answering the user's message. If the current message is self-contained, do not rely on previous context.

Rewritten Specific Query:""")
        
        # Enhanced QA prompt with examples
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Context from knowledge base:
{context}
User Query: {question}
Chat History:
{chat_history}

Instruction:
Use chat history only when it is necessary for understanding or answering the user's message. If the current message is self-contained, do not rely on previous context.
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
