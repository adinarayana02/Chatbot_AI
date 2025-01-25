import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from typing import List
import functools
import abc

# Load environment variables (API keys)
load_dotenv()

# Define constants
CONVO_TRAIL_CUTOFF = 5
PERSONAL_AI_ASSISTANT_PROMPT_HEAD = "You are a helpful assistant. [[previous_interactions]] [[latest_input]]"
ASSISTANT_TYPE = "GroqPAF"

# Define Interaction class
class Interaction:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class PersonalAssistantFramework(abc.ABC):
    @staticmethod
    def timeit_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def think(self, prompt: str) -> str:
        pass

class GroqPAF(PersonalAssistantFramework):
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def setup(self):
        self.llm_model = None

    @PersonalAssistantFramework.timeit_decorator
    def think(self, thought: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",  # You can change this to other Groq models
                messages=[
                    {"role": "system", "content": "You are a helpful assistant named Luna."},
                    {"role": "user", "content": thought}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"

def build_prompt(latest_input: str, previous_interactions: List[Interaction]) -> str:
    previous_interactions_str = "\n".join(
        [
            f"""<interaction>
            <role>{interaction.role}</role>
            <content>{interaction.content}</content>
        </interaction>"""
            for interaction in previous_interactions
        ]
    )
    prepared_prompt = PERSONAL_AI_ASSISTANT_PROMPT_HEAD.replace(
        "[[previous_interactions]]", previous_interactions_str
    ).replace("[[latest_input]]", latest_input)

    return prepared_prompt

# Set the page config for Streamlit
st.set_page_config(page_title="Luna AI Chatbot", page_icon="🤖", layout="wide")

# Add premium CSS styling with the new blue color palette
st.markdown("""
<style>
    .stApp {
        background-color: #003135;  /* Dark blue background for a premium feel */
        color: #AFDDE5;             /* Light blue text for contrast */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Modern font */
    }
    .header {
        background-color: #024950;  /* Medium blue header */
        padding: 20px;
        text-align: center;
        color: #AFDDE5;             /* Light blue text in the header */
        font-size: 28px;
        font-weight: bold;
        border-bottom: 1px solid #0FA4AF; /* Lighter blue border */
    }
    .chat-box {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        max-width: 100%;
        word-wrap: break-word;
        background-color: #024950;  /* Medium blue chat box background */
        color: #AFDDE5;             /* Light blue text for chat box */
    }
    .user-message {
        text-align: right;
        margin-left: auto;
        background-color: #0FA4AF;  /* Aqua blue for user messages */
        color: #003135;             /* Dark blue text for user messages */
    }
    .assistant-message {
        text-align: left;
        margin-right: auto;
        background-color: #003135;  /* Dark blue for assistant messages */
        color: #AFDDE5;             /* Light blue text for assistant messages */
        display: flex;
        align-items: center;
    }
    .assistant-logo {
        width: 40px;
        height: 40px;
        margin-right: 10px;
        border-radius: 50%;
        background-color: #0FA4AF;  /* Aqua blue circle for logo */
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .assistant-logo img {
        width: 24px;
        height: 24px;
    }
    .message {
        display: inline-block;
        padding: 10px;
        border-radius: 15px;
        width: fit-content;
    }
    .stButton button {
        background-color: #0FA4AF;  /* Aqua blue button color */
        color: #003135;            /* Dark blue text on button */
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #024950; /* Medium blue on hover */
    }
    .stTextArea textarea {
        background-color: #024950;  /* Medium blue background for text area */
        color: #AFDDE5;             /* Light blue text for text area */
        border: none;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
        resize: none;
    }
    .stTextArea textarea::placeholder {
        color: #AFDDE5;             /* Light blue placeholder text */
    }
    .subheader {
        color: #AFDDE5;             /* Light blue color for subheaders */
        font-size: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header and container for chat messages
st.markdown('<div class="header">✨Luna✨<br>Your Friendly Digital Assistant chatbot!😊</div>', unsafe_allow_html=True)

# Container for chat messages and input area
with st.container():
    # State to store previous interactions
    if 'previous_interactions' not in st.session_state:
        st.session_state.previous_interactions = []

    # Conversation history
    st.markdown(
        '''
        <div class="subheader">
            Meet Luna,your friendly Ai assistant ready to help you in any language! 🌐💬 
            
        </div>
        ''',
        unsafe_allow_html=True
    )
    for interaction in st.session_state.previous_interactions:
        if interaction.role == "human":
            st.markdown(f'<div class="chat-box user-message message"><strong>You:</strong> {interaction.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-box assistant-message message">
                <div class="assistant-logo">
                    <img src="https://avatar-prod-us-east-2.webexcontent.com/Avtr~V1~1eb65fdf-9643-417f-9974-ad72cae0e10f/V1~1a08968daf428f4f5d1ce0f04de95eb40bf8d6d8cb5a0463a6f858f52e3a9df9~49713ad2dabe4cb2a6f83a05c6d6e4f5"" alt="AI Assistant">
                </div>
                <div><strong>Assistant:</strong> {interaction.content}</div>
            </div>
            """, unsafe_allow_html=True)

    # Text input field
    user_input = st.text_area("Enter your message:", height=100, placeholder="Type your message here...")

    # Submit button
    if st.button("Send"):
        if user_input:
            assistant = GroqPAF()
            assistant.setup()

            # Build prompt and get response
            prompt = build_prompt(user_input, st.session_state.previous_interactions)
            response = assistant.think(prompt)

            # Update interactions
            st.session_state.previous_interactions.append(Interaction(role="human", content=user_input))
            st.session_state.previous_interactions.append(Interaction(role="assistant", content=response))

            # Display response
            st.markdown(f'<div class="chat-box user-message message"><strong>You:</strong> {user_input}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="chat-box assistant-message message">
                <div class="assistant-logo">
                    <img src="https://avatar-prod-us-east-2.webexcontent.com/Avtr~V1~1eb65fdf-9643-417f-9974-ad72cae0e10f/V1~1a08968daf428f4f5d1ce0f04de95eb40bf8d6d8cb5a0463a6f858f52e3a9df9~49713ad2dabe4cb2a6f83a05c6d6e4f5" alt="AI Assistant">
                </div>
                <div><strong>Assistant:</strong> {response}</div>
            </div>
            """, unsafe_allow_html=True)

            # Keep only the last CONVO_TRAIL_CUTOFF interactions
            if len(st.session_state.previous_interactions) > CONVO_TRAIL_CUTOFF:
                st.session_state.previous_interactions = st.session_state.previous_interactions[-CONVO_TRAIL_CUTOFF:]

        else:
            st.warning("Please enter a message.")

st.write("This chatbot uses Groq's Mixtral 8x7B model to respond to your messages.")
