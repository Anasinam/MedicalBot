import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
import os
import json

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set Google credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Chatbot/MediGuide/clever-grammar-438917-m3-3994185b2e58.json"

# Folder to store chat history
CHAT_HISTORY_FOLDER = "chat_history"
os.makedirs(CHAT_HISTORY_FOLDER, exist_ok=True)

# Set page config and title
st.set_page_config(page_title="MediGuide", page_icon=":speech_balloon:", layout="wide")

# Sidebar CSS styling
st.markdown(
    """
    <style>
        /* Allow Sidebar Resizing */
        [data-testid="stSidebar"] {
            width: auto !important; /* Enable flexible width */
            min-width: 320px !important; /* Minimum width */
            max-width: 350px !important; /* Maximum width */
            resize: horizontal; /* Allow horizontal resizing */
            overflow: auto; /* Enable scrolling if content overflows */
            padding: 20px;
            background-color: #f6f6f7; /* Ghost White background */
            color: #1B1B20; /* Black Russian text */
            border-right: 1px solid #c1cce1; /* Subtle separator border */
        }

        /* Prevent Sidebar Content Overlap */
        .block-container {
            margin-left: 10px; /* Set margin for the main container */
            transition: margin-left 0.2s; /* Smooth transition for resizing */
        }

        /* Adjust content to avoid overlap with fixed sidebar */
        .block-container {
            padding-left: 320px !important; /* Push main content to account for fixed sidebar width */
        }

        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #f6f6f7; /* Ghost White background */
            color: #1B1B20; /* Black Russian text */
            padding: 20px;
            border-right: 1px solid #c1cce1; /* Subtle separator border */
        }

        /* Sidebar Header Styling 
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #1B1B20; /* black Russian text for headers */
            font-family: Arial, sans-serif;
            font-size: 12px;
            font-weight: normal;
            margin-bottom: 12px;
            text-align: left;
        }

        .sidebar-title img {
            width: 2px; /* Icon size */
            height: 5px;
            margin-right: 5px; 
        }

        /* Buttons Styling */
        [data-testid="stSidebar"] button {
            background-color: #ebe8e8 !important; /* Ghost White backgorund color */
            color: #1B1B20 !important; /* Solitude color */
            border: none;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            padding: 8px 10px;
            margin: -3px 0;
            transition: all 0.3s ease;
        }

        [data-testid="stSidebar"] button:hover {
            background-color: #e1e1e3 !important; /* Bright Grey color for buttons */
            transform: scale(1.02); /* Subtle hover effect 
        }

        /* Saved Chats Buttons */
        [data-testid="stSidebar"] .stButton>button {
            background-color: #ffffff !important;
            color: #000000 !important;
            font-size: 5px;
            margin-top: 2px;
            transition: all 0.2s ease;
        }

        /* Sidebar Links/Labels */
        [data-testid="stSidebar"] label {
            color: #e1dcdc !important; 
            font-size: 10px;
            font-weight: 500;
            margin-bottom: 5px;
        }
        
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the alignment manually
horizontal_alignment = 30  # Horizontal alignment: 0 = Left, 50 = Center, 100 = Right
vertical_alignment = 20    # Vertical alignment: 0 = Top, 50 = Center, 100 = Bottom

st.markdown(
    f"""
    <style>
        .manual-alignment-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: left;
            height: 25vh; /* Full viewport height */
            position: relative;
            transform: translate({horizontal_alignment - 50}%, {vertical_alignment - 50}%); 
            text-align: center;
        }}
        .manual-icon {{
            width: 100px; /* Adjust icon size */
            margin-bottom: 10px; /* Spacing below icon */
        }}
        .manual-title {{
            font-size: 36px;
            font-weight: bold;
            color: #1F2937;
            margin: 0;
        }}
        .manual-subtitle {{
            font-size: 18px;
            color: #4B5563;
            margin-top: 10px;
        }}
    </style>
    <div class="manual-alignment-container">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" class="manual-icon" alt="MediGuide Icon">
        <h1 class="manual-title">MediGuide 💬</h1>
        <p class="manual-subtitle">Welcome to MediGuide – Your Virtual Health Assistant!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

if 'loading' not in st.session_state:
    st.session_state.loading = False

if 'current_chat_file' not in st.session_state:
    st.session_state.current_chat_file = None

# Helper functions for chat history
def get_current_chat_file():
    """Generate or retrieve the current session's chat file name based on the first user query."""
    if st.session_state.chat_history and not st.session_state.current_chat_file:
        # Extract first query
        first_question = st.session_state.chat_history[0][1]
        # Clean and shorten the first query for filename
        cleaned_name = re.sub(r'[^\w\s]', '', first_question)  # Remove special characters
        cleaned_name = re.sub(r'\s+', '_', cleaned_name)  # Replace spaces with underscores
        cleaned_name = cleaned_name[:20]  # Limit filename to 20 characters

        # Add date for uniqueness
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        st.session_state.current_chat_file = os.path.join(
            CHAT_HISTORY_FOLDER, f"{date_prefix}_{cleaned_name}.json"
        )
    elif not st.session_state.current_chat_file:
        # Fallback to timestamp if no user input is found
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.current_chat_file = os.path.join(CHAT_HISTORY_FOLDER, f"chat_{timestamp}.json")
    return st.session_state.current_chat_file

def refresh_sidebar():
    """Trigger a refresh by rerunning the script."""
    st.rerun()

def save_chat_to_file():
    file_path = get_current_chat_file()
    formatted_chat = [{"role": role, "content": message} for role, message in st.session_state.chat_history]
    with open(file_path, "w") as file:
        json.dump(formatted_chat, file, indent=4)

def list_saved_chats():
    return [f.replace(".json", "") for f in os.listdir(CHAT_HISTORY_FOLDER) if f.endswith(".json")]

def load_chat_from_file(session_name):
    file_path = os.path.join(CHAT_HISTORY_FOLDER, f"{session_name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return []

def list_saved_chats_grouped():
    saved_chats = [f.replace(".json", "") for f in os.listdir(CHAT_HISTORY_FOLDER) if f.endswith(".json")]
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    grouped_chats = {"Today": [], "Yesterday": [], "Other": {}}

    for chat in saved_chats:
        match = re.search(r"(\d{4}-\d{2}-\d{2})_(.*)", chat)
        if match:
            chat_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            chat_name = match.group(2)
            if chat_date == today:
                grouped_chats["Today"].append((chat, chat_name))
            elif chat_date == yesterday:
                grouped_chats["Yesterday"].append((chat, chat_name))
            else:
                date_str = chat_date.strftime("%Y-%m-%d")
                if date_str not in grouped_chats["Other"]:
                    grouped_chats["Other"][date_str] = []
                grouped_chats["Other"][date_str].append((chat, chat_name))
    return grouped_chats

def display_sidebar_chat_history_grouped():
    """Display the grouped chat history with embedded delete icons."""
    grouped_chats = list_saved_chats_grouped()

    # Custom CSS for icon buttons
    st.markdown(
        """
        <style>
        .icon-button {
            background: none;
            border: none;
            cursor: pointer;
            margin: 0;
            padding: 5px;
            font-size: 16px;
        }
        .icon-button:hover {
            color: #ff4b4b; /* Red for delete hover */
        }
        /* Bold Styling for Today, Yesterday and Other */
        .section-title {
            font-weight: bold;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def refresh_state():
        """Trigger a refresh by resetting the state."""
        st.experimental_rerun()

    def render_saved_chat(chat_filename, chat_name):
        # Replace underscores with spaces for display
        display_name = chat_name.replace("_", " ")  # Replace underscores with spaces
        display_name = (display_name[:20] + "...") if len(display_name) > 20 else display_name  # Truncate if name is too long

        col1, col2 = st.columns([0.7, 0.15,])  # Layout: Chat name, Delete
        with col1:
            if st.button(display_name, key=f"load-{chat_filename}"):  # Use display_name for the button
                chat_data = load_chat_from_file(chat_filename)  # Load the selected chat
                st.session_state.chat_history = [(item["role"], item["content"]) for item in chat_data]  # Update session state
                st.session_state.current_chat_file = chat_filename  # Set the current chat file
                st.success(f"Loaded chat: {display_name}")  # Notify user
        
        with col2:
            delete_icon = "🗑️"  # Trash can icon for delete
            if st.button(delete_icon, key=f"delete-{chat_filename}", help="Delete this chat", use_container_width=True):
                file_path = os.path.join(CHAT_HISTORY_FOLDER, f"{chat_filename}.json")
                try:
                    os.remove(file_path)
                    st.success(f"Deleted chat '{display_name}'")
                    refresh_sidebar()
                except Exception as e:
                    st.error(f"Error deleting chat: {e}")

    # Add CSS to prevent wrapping and truncate text
    st.markdown(
        """
        <style>
        .stButton>button {
            text-overflow: ellipsis;
            white-space: nowrap;
            overflow: hidden;
            display: inline-block;
            max-width: 100%; /* Adjust as needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # "Today" Section
    if grouped_chats["Today"]:
        st.markdown("<div class='section-title'>Today</div>", unsafe_allow_html=True)
        for chat_filename, chat_name in grouped_chats["Today"]:
            render_saved_chat(chat_filename, chat_name)

    # "Yesterday" Section
    if grouped_chats["Yesterday"]:
        st.markdown("<div class='section-title'>Yesterday</div>", unsafe_allow_html=True)
        for chat_filename, chat_name in grouped_chats["Yesterday"]:
            render_saved_chat(chat_filename, chat_name)

    # Other Date Sections
    if grouped_chats["Other"]:
        st.markdown("<div class='section-title'>Other</div>", unsafe_allow_html=True)
        for date, chats in grouped_chats["Other"].items():
            st.subheader(date)  # Keep this as subheader for normal appearance
            for chat_filename, chat_name in chats:
                render_saved_chat(chat_filename, chat_name)

# Function to load the vector store
def load_vector_store(vector_store_path="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available, 
    say, "Sorry I am not able to answer this, please consult Doctor for this".\n\n
    Context:\n {context}\n
    Question: {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def initialize_vector_store():
#     """Load the pre-trained vector store."""
#     try:
#         load_vector_store()  # Ensure this function points to your faiss_index loading logic
#         st.session_state.vector_store_initialized = True
#         print("Vector store initialized successfully.")
#     except Exception as e:
#         st.error("Error loading vector store. Ensure it has been created and is accessible.")
#         print(f"Error: {e}")

def initialize_vector_store():
    """Load the pre-trained vector store."""
    try:
        load_vector_store()  # Ensure this function points to your faiss_index loading logic
        st.session_state.vector_store_initialized = True
        print("Vector store initialized successfully.")
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        print(f"Error: {e}")

def user_input():
    user_question = st.session_state.user_input

    # Ensure vector store is initialized
    if not st.session_state.vector_store_initialized:
        st.error("Vector store not loaded. Please ensure it is trained and accessible.")
        return

    st.session_state.loading = True

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("C:/Chatbot/MediGuide/faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.session_state.chat_history.append(("human", user_question))
        st.session_state.chat_history.append(("AI", response["output_text"]))

        save_chat_to_file()
    except Exception as e:
        st.error("Error generating response. Please check the vector store or embeddings.")
        print(f"Error: {e}")
    finally:
        st.session_state.loading = False
        st.session_state.user_input = ""


def display_chat_history():
    st.markdown(
         """
        <style>
            .input-container {
                position: fixed; /* Fixed at bottom */
                bottom: 10;
                left: 0;
                width: -20%;
                padding: 10px 50px;
                background-color: #1e1e1e; /* Slightly dark gray */
                box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.2);
                z-index: 999; /* Make sure input is above everything */
                border-radius: 10px;
            }
            .chat-container {
                display: flex;
                flex-direction: column;
                align-items: flex-start; /* Align all chat content to the left */
                padding: 10px;
                margin-left: 20px; /* Adjust spacing from the edge */
                background-color: #f6f6f7; /* Light gray background for the bot area */
                border-radius: 5px; /* Rounded edges for aesthetics */
            }
            .chat-message {
                max-width: 70%; /* Restrict message width to 70% of the container */
                padding: 10px;
                margin: 5px 0; /* Spacing between messages */
                border-radius: 10px;
                font-size: 14px;
                line-height: 1.5;
                word-wrap: break-word;
            }
            .bot-message {
                background-color: #e0e0e0; /* Light gray for bot messages */
                color: #000;
                align-self: flex-start;
            }
            .user-message {
                background-color: #0084ff;
                color: #fff;
                align-self: flex-end;
            }
    </style>
    """,
    unsafe_allow_html=True,
)
    
    for role, message in st.session_state.chat_history:
        if role == "human":
            st.markdown(f"<div class='chat-message user-message'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'>{message}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    if not st.session_state.vector_store_initialized:
        initialize_vector_store()  # Load the vector store at app startup

    with st.sidebar:
        # MediGuide Header with Icon
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 10px;">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" 
                    alt="Chatbot Icon" 
                    style="width: 80px; height: 80px; margin-bottom: 5px;">
                <h2 style="color: #1B1B20; font-size: 20px; font-weight: bold;">MediGuide</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Grouped Saved Chats
        display_sidebar_chat_history_grouped()

    display_chat_history()

    # Input box fixed at the bottom
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    st.text_input("Type your question...", key="user_input", on_change=user_input, placeholder="Message MediGuide")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.loading:
        st.markdown("🔄 Generating response...")

if __name__ == "__main__":
    main()
