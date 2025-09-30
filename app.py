import streamlit as st
import time
import json
from datetime import datetime
import google.generativeai as genai
import os

# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Configure Gemini API - Use Streamlit secrets for deployment
try:
    # For Streamlit Cloud deployment, use secrets
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    
    # For local development, fall back to environment variable
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        genai.configure(api_key=api_key)
        API_CONFIGURED = True
    else:
        API_CONFIGURED = False
except Exception as e:
    API_CONFIGURED = False
    st.error(f"API Configuration Error: {str(e)}")

# Inject custom CSS for a modern, minimal frontend
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* General Styling */
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main App Background */
    .stApp {
        background-color: #111827;
    }

    /* Chat Message Styling */
    [data-testid="stChatMessage"] {
        padding: 1.25rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid transparent;
        transition: all 0.2s ease-in-out;
    }

    /* Assistant (Bot) Message Style */
    div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #374151;
        border-left: 3px solid #10b981;
    }

    /* User Message Style */
    div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
        background-color: #4f46e5;
        border-left: 3px solid #818cf8;
    }

    /* Text color inside messages */
    [data-testid="stChatMessage"] p {
        color: #F9FAFB;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* App Title */
    h1 {
        text-align: center;
        color: #F9FAFB;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1F2937;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #F9FAFB;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: #D1D5DB;
    }

    /* Chat Input Box */
    [data-testid="stChatInput"] {
        background-color: #1F2937;
        border-top: 1px solid #374151;
        border-radius: 1rem;
    }
    
    /* FAQ Button Styles */
    .stButton > button {
        background-color: #374151;
        color: #F9FAFB;
        border: 1px solid #4B5563;
        border-radius: 0.75rem;
        padding: 0.5rem 0.75rem;
        font-size: 14px;
        transition: all 0.2s ease-in-out;
        font-weight: 400;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #4B5563;
        border-color: #6B7280;
        color: #ffffff;
        transform: translateY(-1px);
    }
    
    .stButton > button:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    /* Helper text for FAQs */
    .faq-header {
        width: 100%;
        text-align: center;
        color: #9CA3AF;
        margin-bottom: 1rem;
        margin-top: 1rem;
        font-size: 14px;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #10b981;
    }

    /* Success/Info boxes */
    .stAlert {
        background-color: #1F2937;
        border-left: 3px solid #10b981;
        color: #F9FAFB;
    }

</style>
""", unsafe_allow_html=True)

# System prompts for different personalities
PERSONALITY_PROMPTS = {
    "friendly": "You are a helpful, friendly, and warm AI assistant. Use emojis occasionally and be conversational. Keep responses concise but informative.",
    "professional": "You are a professional AI assistant. Provide clear, concise, and formal responses. Maintain a business-like tone.",
    "casual": "You are a casual, laid-back AI assistant. Chat like a friend - be relaxed, use casual language, and keep things fun and light.",
    "creative": "You are a creative and imaginative AI assistant. Think outside the box, use metaphors, and make your responses engaging and colorful.",
    "technical": "You are a technical AI assistant with expertise in technology and programming. Provide detailed, accurate technical information when relevant.",
}

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Session 1": []}
    st.session_state.current_session = "Session 1"

if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini-flash-latest"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2000

if "personality" not in st.session_state:
    st.session_state.personality = "friendly"

if "show_timestamps" not in st.session_state:
    st.session_state.show_timestamps = False

if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True

if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

def initialize_chat_model():
    """Initialize or reinitialize the chat model with current settings."""
    try:
        generation_config = {
            "temperature": st.session_state.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": st.session_state.max_tokens,
        }
        
        system_prompt = PERSONALITY_PROMPTS[st.session_state.personality]
        
        model = genai.GenerativeModel(
            model_name=st.session_state.model_name,
            generation_config=generation_config,
            system_instruction=system_prompt
        )
        
        # Convert existing messages to Gemini format for context
        history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        
        chat = model.start_chat(history=history)
        st.session_state.chat_model = chat
        return True
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return False

def get_gemini_response(user_input, streaming=True):
    """Get response from Gemini API."""
    if streaming:
        response = st.session_state.chat_model.send_message(user_input, stream=True)
        return response
    else:
        response = st.session_state.chat_model.send_message(user_input)
        return response.text

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Chat Settings")
    
    # API Status
    if API_CONFIGURED:
        st.success("✅ AI Powered & Ready")
    else:
        st.error("❌ API Not Configured")
        st.info("💡 Developer: Add GEMINI_API_KEY to Streamlit secrets or environment variables")
    
    st.divider()
    
    # Model Configuration (only if API is configured)
    if API_CONFIGURED:
        st.subheader("🤖 AI Settings")
        
        personality = st.selectbox(
            "Bot Personality",
            list(PERSONALITY_PROMPTS.keys()),
            index=list(PERSONALITY_PROMPTS.keys()).index(st.session_state.personality),
            format_func=lambda x: x.title()
        )
        
        with st.expander("⚙️ Advanced Settings"):
            if st.session_state.model_name not in ["gemini-flash-latest", "gemini-pro-latest", "gemini-pro"]:
                st.session_state.model_name = "gemini-flash-latest"
            model_name = st.selectbox(
                "Model",
                ["gemini-flash-latest", "gemini-pro-latest", "gemini-pro"],
                index=["gemini-flash-latest", "gemini-pro-latest", "gemini-pro"].index(st.session_state.model_name),
                help="Flash is faster, Pro is more capable"
            )
            
            temperature = st.slider(
                "Creativity",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
            
            max_tokens = st.slider(
                "Response Length",
                min_value=500,
                max_value=4000,
                value=st.session_state.max_tokens,
                step=500,
                help="Maximum length of AI responses"
            )
            
            streaming = st.checkbox(
                "Streaming Responses",
                value=st.session_state.streaming_enabled,
                help="See responses as they're generated"
            )
        
        # Apply settings
        settings_changed = (
            personality != st.session_state.personality or
            model_name != st.session_state.model_name or
            temperature != st.session_state.temperature or
            max_tokens != st.session_state.max_tokens
        )
        
        if settings_changed:
            if st.button("✨ Apply Changes", use_container_width=True):
                st.session_state.personality = personality
                st.session_state.model_name = model_name
                st.session_state.temperature = temperature
                st.session_state.max_tokens = max_tokens
                st.session_state.streaming_enabled = streaming
                
                if initialize_chat_model():
                    st.success("Settings updated! ✅")
                    time.sleep(0.5)
                    st.rerun()
        
        st.divider()
        
        # Session Management
        st.subheader("💬 Sessions")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            session_options = list(st.session_state.chat_sessions.keys())
            selected_session = st.selectbox(
                "Current Session",
                session_options,
                index=session_options.index(st.session_state.current_session),
                label_visibility="collapsed"
            )
        
        with col2:
            if st.button("➕", use_container_width=True, help="New Session"):
                session_num = len(st.session_state.chat_sessions) + 1
                new_session = f"Session {session_num}"
                st.session_state.chat_sessions[new_session] = []
                st.session_state.current_session = new_session
                st.session_state.messages = []
                initialize_chat_model()
                st.rerun()
        
        if selected_session != st.session_state.current_session:
            st.session_state.current_session = selected_session
            st.session_state.messages = st.session_state.chat_sessions[selected_session]
            initialize_chat_model()
            st.rerun()
        
        st.divider()
        
        # Display Options
        st.subheader("🎨 Display")
        
        show_timestamps = st.checkbox(
            "Show Timestamps",
            value=st.session_state.show_timestamps
        )
        st.session_state.show_timestamps = show_timestamps
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Stats")
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("You", user_msgs, delta=None)
        with col2:
            st.metric("AI", bot_msgs, delta=None)
        
        st.divider()
        
        # Actions
        st.subheader("🔧 Actions")
        
        if st.button("📥 Export Chat", use_container_width=True):
            if st.session_state.messages:
                chat_export = {
                    "session": st.session_state.current_session,
                    "exported_at": datetime.now().isoformat(),
                    "model": st.session_state.model_name,
                    "personality": st.session_state.personality,
                    "messages": st.session_state.messages
                }
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(chat_export, indent=2),
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No messages to export")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_sessions[st.session_state.current_session] = []
            initialize_chat_model()
            st.rerun()
        
        st.divider()
        st.caption("Powered by Google Gemini 🤖")

# Main chat area
st.title("🤖 AI Chatbot")

if not API_CONFIGURED:
    st.error("⚠️ Chatbot is not configured. Please contact the administrator.")
    st.stop()

st.caption(f"Personality: **{st.session_state.personality.title()}** • Model: **{st.session_state.model_name}**")

# Initialize chat model if not already done
if st.session_state.chat_model is None:
    initialize_chat_model()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if st.session_state.show_timestamps and "timestamp" in message:
            st.caption(f"🕒 {message['timestamp']}")

# FAQ Section (only when chat is empty)
clicked_prompt = None
if len(st.session_state.messages) == 0:
    st.markdown('<p class="faq-header">✨ Start a conversation</p>', unsafe_allow_html=True)

    faq_questions = {
        "👋 Say Hello": "Hi! Tell me about yourself.",
        "💡 Get Ideas": "Give me a creative idea",
        "📚 Learn Something": "Teach me something interesting",
        "✍️ Be Creative": "Write me a short poem"
    }

    cols = st.columns(len(faq_questions))
    for i, (display_text, prompt_text) in enumerate(faq_questions.items()):
        with cols[i]:
            if st.button(display_text, key=f"faq_{i}", use_container_width=True):
                clicked_prompt = prompt_text

# Handle user input
prompt_from_input = st.chat_input("Type your message here...")
final_prompt = clicked_prompt if clicked_prompt else prompt_from_input

if final_prompt:
    timestamp = datetime.now().strftime("%I:%M %p")
    
    # Add user message
    user_message = {
        "role": "user",
        "content": final_prompt,
        "timestamp": timestamp
    }
    st.session_state.messages.append(user_message)
    st.session_state.chat_sessions[st.session_state.current_session].append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(final_prompt)
        if st.session_state.show_timestamps:
            st.caption(f"🕒 {timestamp}")

    # Get and display AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if st.session_state.streaming_enabled:
                response_stream = get_gemini_response(final_prompt, streaming=True)
                
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            else:
                with st.spinner("🤔 Thinking..."):
                    full_response = get_gemini_response(final_prompt, streaming=False)
                message_placeholder.markdown(full_response)
        
        except Exception as e:
            full_response = f"❌ Sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(full_response)
        
        bot_timestamp = datetime.now().strftime("%I:%M %p")
        if st.session_state.show_timestamps:
            st.caption(f"🕒 {bot_timestamp}")
    
    # Save AI response
    bot_message = {
        "role": "assistant",
        "content": full_response,
        "timestamp": bot_timestamp
    }
    st.session_state.messages.append(bot_message)
    st.session_state.chat_sessions[st.session_state.current_session].append(bot_message)
