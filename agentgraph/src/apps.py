import streamlit as st
from chatbot.chatbot_backend import ChatBot
from utils.ui_settings import UISettings
import os
import yaml
from pyprojroot import here
from prepare_vector_db import PrepareVectorDB
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Helper functions for vector database creation
def create_swiss_database(app_config):
    """Create Swiss Airline Policy vector database"""
    with st.spinner("Creating Swiss Airline Policy database..."):
        try:
            # Extract configuration
            chunk_size = app_config["swiss_airline_policy_rag"]["chunk_size"]
            chunk_overlap = app_config["swiss_airline_policy_rag"]["chunk_overlap"]
            embedding_model = app_config["swiss_airline_policy_rag"]["embedding_model"]
            vectordb_dir = app_config["swiss_airline_policy_rag"]["vectordb"]
            collection_name = app_config["swiss_airline_policy_rag"]["collection_name"]
            doc_dir = app_config["swiss_airline_policy_rag"]["unstructured_docs"]
            
            # Create progress placeholder
            progress_placeholder = st.empty()
            
            # Create instance and run
            prepare_db_instance = PrepareVectorDB(
                doc_dir=doc_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                vectordb_dir=vectordb_dir,
                collection_name=collection_name
            )
            
            progress_placeholder.info("Processing documents and creating embeddings...")
            prepare_db_instance.run()
            
            progress_placeholder.success("✅ Swiss Airline Policy database created successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error creating Swiss database: {str(e)}")

def create_stories_database(app_config):
    """Create Stories vector database"""
    with st.spinner("Creating Stories database..."):
        try:
            # Extract configuration
            chunk_size = app_config["stories_rag"]["chunk_size"]
            chunk_overlap = app_config["stories_rag"]["chunk_overlap"]
            embedding_model = app_config["stories_rag"]["embedding_model"]
            vectordb_dir = app_config["stories_rag"]["vectordb"]
            collection_name = app_config["stories_rag"]["collection_name"]
            doc_dir = app_config["stories_rag"]["unstructured_docs"]
            
            # Create progress placeholder
            progress_placeholder = st.empty()
            
            # Create instance and run
            prepare_db_instance = PrepareVectorDB(
                doc_dir=doc_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                vectordb_dir=vectordb_dir,
                collection_name=collection_name
            )
            
            progress_placeholder.info("Processing documents and creating embeddings...")
            prepare_db_instance.run()
            
            progress_placeholder.success("✅ Stories database created successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error creating Stories database: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="AgentGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gradio_chat_history" not in st.session_state:
    st.session_state.gradio_chat_history = []

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    .bot-message {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🤖 AgentGraph</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.gradio_chat_history = []
        st.rerun()
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ Information")
    st.markdown("""
    **AgentGraph** is an AI assistant that can:
    - Answer general questions
    - Search the web for current information
    - Query databases (Chinook, Travel)
    - Look up Swiss airline policies
    - Search story collections
    
    Simply type your question below and press Enter!
    """)
    
    # Statistics
    if st.session_state.messages:
        st.markdown("---")
        st.subheader("📊 Session Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Conversations", len(st.session_state.messages) // 2)
    
    # Vector Database Management
    st.markdown("---")
    st.subheader("🗄️ Vector Database")
    
    # Load configuration
    try:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        config_loaded = True
    except Exception as e:
        st.error(f"Could not load config: {str(e)}")
        config_loaded = False
    
    if config_loaded:
        # Check API key
        api_key_available = bool(os.getenv('OPENAI_API_KEY'))
        
        if not api_key_available:
            st.error("⚠️ OPENAI_API_KEY not found in environment variables")
            st.info("Please set your API key in the .env file to create databases")
        
        # Swiss Airline Policy Database
        st.markdown("**Swiss Airline Policy**")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            swiss_db_path = app_config["swiss_airline_policy_rag"]["vectordb"]
            if os.path.exists(here(swiss_db_path)):
                st.success("✅ Ready")
            else:
                st.warning("⚠️ Not created")
        
        with col2:
            if st.button("🔄", key="swiss_create", help="Create Swiss Airline Policy Database", disabled=not api_key_available):
                create_swiss_database(app_config)
        
        # Stories Database
        st.markdown("**Stories Database**")
        col3, col4 = st.columns([3, 1])
        
        with col3:
            stories_db_path = app_config["stories_rag"]["vectordb"]
            if os.path.exists(here(stories_db_path)):
                st.success("✅ Ready")
            else:
                st.warning("⚠️ Not created")
        
        with col4:
            if st.button("🔄", key="stories_create", help="Create Stories Database", disabled=not api_key_available):
                create_stories_database(app_config)
        
        # Create All Databases button
        st.markdown("---")
        if st.button("🚀 Create All Databases", use_container_width=True, disabled=not api_key_available):
            with st.spinner("Creating all databases..."):
                create_swiss_database(app_config)
                create_stories_database(app_config)

# Main interface with tabs
tab1, tab2 = st.tabs(["💬 Chat Interface", "🗄️ Vector Database Manager"])

with tab1:
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter your message here..."):
        # Add user message to chat history
        user_avatar = "images/AI_RT.png" if os.path.exists("images/AI_RT.png") else "🧑‍💻"
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "avatar": user_avatar
        })
        
        # Display user message
        with st.chat_message("user", avatar=user_avatar):
            st.write(prompt)
        
        # Process the message through ChatBot
        with st.spinner("Thinking..."):
            try:
                # Call the ChatBot.respond function
                # Convert current messages to gradio format for compatibility
                gradio_format = [(msg["content"], None) for msg in st.session_state.messages if msg["role"] == "user"]
                if len(gradio_format) > len(st.session_state.gradio_chat_history):
                    # Add the new message
                    st.session_state.gradio_chat_history = gradio_format[:-1]  # All but the last
                
                # Get response from ChatBot
                _, updated_gradio_history = ChatBot.respond(st.session_state.gradio_chat_history, prompt)
                
                # Extract the latest bot response
                if updated_gradio_history and len(updated_gradio_history) > 0:
                    latest_response = updated_gradio_history[-1][1]  # Get the bot's response
                    
                    # Add bot response to chat history
                    bot_avatar = "images/openai.png" if os.path.exists("images/openai.png") else "🤖"
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": latest_response,
                        "avatar": bot_avatar
                    })
                    
                    # Update gradio chat history for next iteration
                    st.session_state.gradio_chat_history = updated_gradio_history
                    
                    # Display bot response
                    with st.chat_message("assistant", avatar=bot_avatar):
                        st.write(latest_response)
                else:
                    # Fallback response
                    error_msg = "I'm sorry, I couldn't process your request at the moment."
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "avatar": "🤖"
                    })
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(error_msg)
                        
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "avatar": "⚠️"
                })
                with st.chat_message("assistant", avatar="⚠️"):
                    st.error(error_msg)

with tab2:
    st.subheader("🗄️ Vector Database Manager")
    
    # Load configuration for detailed view
    try:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        config_loaded = True
    except Exception as e:
        st.error(f"Could not load config: {str(e)}")
        config_loaded = False
    
    if config_loaded:
        # API Key Status
        api_key_available = bool(os.getenv('OPENAI_API_KEY'))
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if api_key_available:
                st.success("🔑 API Key: Ready")
            else:
                st.error("🔑 API Key: Missing")
        
        with col2:
            if not api_key_available:
                st.warning("Please set OPENAI_API_KEY in your .env file to create vector databases")
        
        st.markdown("---")
        
        # Database Information
        st.subheader("📊 Database Status")
        
        # Swiss Airline Policy Database
        with st.expander("🛫 Swiss Airline Policy Database", expanded=True):
            swiss_config = app_config["swiss_airline_policy_rag"]
            swiss_db_path = swiss_config["vectordb"]
            swiss_exists = os.path.exists(here(swiss_db_path))
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Status:** {'✅ Created' if swiss_exists else '❌ Not Created'}")
                st.write(f"**Path:** `{swiss_db_path}`")
                st.write(f"**Documents:** `{swiss_config['unstructured_docs']}`")
                st.write(f"**Collection:** `{swiss_config['collection_name']}`")
            
            with col2:
                st.write(f"**Chunk Size:** {swiss_config['chunk_size']}")
                st.write(f"**Overlap:** {swiss_config['chunk_overlap']}")
                st.write(f"**Model:** {swiss_config['embedding_model']}")
            
            with col3:
                if st.button("🔄 Create/Recreate", key="swiss_detail_create", disabled=not api_key_available):
                    create_swiss_database(app_config)
        
        # Stories Database
        with st.expander("📚 Stories Database", expanded=True):
            stories_config = app_config["stories_rag"]
            stories_db_path = stories_config["vectordb"]
            stories_exists = os.path.exists(here(stories_db_path))
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Status:** {'✅ Created' if stories_exists else '❌ Not Created'}")
                st.write(f"**Path:** `{stories_db_path}`")
                st.write(f"**Documents:** `{stories_config['unstructured_docs']}`")
                st.write(f"**Collection:** `{stories_config['collection_name']}`")
            
            with col2:
                st.write(f"**Chunk Size:** {stories_config['chunk_size']}")
                st.write(f"**Overlap:** {stories_config['chunk_overlap']}")
                st.write(f"**Model:** {stories_config['embedding_model']}")
            
            with col3:
                if st.button("🔄 Create/Recreate", key="stories_detail_create", disabled=not api_key_available):
                    create_stories_database(app_config)
        
        st.markdown("---")
        
        # Bulk Actions
        st.subheader("🚀 Bulk Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Create All Databases", use_container_width=True, disabled=not api_key_available):
                with st.spinner("Creating all databases..."):
                    create_swiss_database(app_config)
                    create_stories_database(app_config)
        
        with col2:
            if st.button("🗑️ Clear Database Cache", use_container_width=True):
                st.info("This would clear cached embeddings (functionality can be added)")
        
        # Instructions
        st.markdown("---")
        st.subheader("📋 Instructions")
        st.markdown("""
        **Vector Database Management:**
        
        1. **Ensure API Key**: Set your `OPENAI_API_KEY` in the `.env` file
        2. **Check Document Sources**: Make sure PDF documents are in the specified directories
        3. **Create Databases**: Click the create buttons to generate vector embeddings
        4. **Monitor Progress**: Watch the progress indicators during creation
        5. **Verify Creation**: Check that databases show "✅ Created" status
        
        **What happens during creation:**
        - PDFs are loaded from the source directories
        - Text is split into chunks based on the configuration
        - OpenAI embeddings are generated for each chunk
        - Vector database is created and saved to disk
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("**AgentGraph** - Powered by LangGraph & OpenAI", help="Advanced AI Assistant with multiple capabilities")

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True) 