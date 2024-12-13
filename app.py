import streamlit as st
from tools import SimpleRAG  # Ensure projectrag.py is in the same folder
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Initialize RAG system
rag = SimpleRAG()
rag.load_documents()

# Streamlit configuration
st.set_page_config(page_title="Saudi Travel Planner", page_icon=":robot_face:")
st.title("Saudi Travel Planner")
st.write("Welcome to the Saudi Travel Planner! \n This app helps you create personalized travel itineraries for exploring Saudi Arabia. \n How to Use: \n Enter Your Query \n Example: Create 2 days itinerary in Riyadh.")

# Add an image
image_path = '/Users/fatimaessa/Downloads/RAG Project/banyan-tree-alula.jpg'  # Replace with your image path
image = Image.open(image_path)  # Load the image
st.image(image, caption='Explore the Beauty of Saudi Arabia', use_container_width=True)  # Display the image

# Custom CSS for the defined theme
st.markdown(
    """
    <style>
        /* General App Styles */
        .stApp {
            background-color: #FFFFFF;  /* White background */
            color: #000000;  /* Black text */
            font-family: 'Arial', sans-serif;  /* Use a clean font */
            line-height: 1.6;  /* Increase line height for better readability */
        }

        /* Sidebar Styles */
        .sidebar .sidebar-content {
            background-color: #007BFF;  /* Blue sidebar */
            color: #FFFFFF;  /* White text in sidebar */
            padding: 20px;  /* Add padding for better spacing */
        }

        /* Input Field Styles */
        .stTextInput>div>input {
            background-color: #F0F0F0;  /* Light gray input field */
            color: #000000;  /* Black text in input field */
            height: 40px;  /* Increase height for better usability */
            border-radius: 5px;  /* Rounded corners */
            padding: 10px;  /* Add padding inside the input */
            border: 1px solid #007BFF;  /* Blue border */
        }
        .stTextInput>div>input:focus {
            border: 2px solid #007BFF;  /* Blue border on focus */
            outline: none;  /* Remove default outline */
        }

        /* Button Styles */
        .stButton>button {
            background-color: #007BFF;  /* Blue button */
            color: #FFFFFF;  /* White text on button */
            padding: 10px 20px;  /* Add padding for larger button */
            border-radius: 5px;  /* Rounded corners */
            font-size: 16px;  /* Increase font size */
            transition: background-color 0.3s;  /* Smooth transition */
        }
        .stButton>button:hover {
            background-color: #0056b3;  /* Darker blue on hover */
        }

        /* Footer Styles */
        footer {
            color: #000000;  /* Black footer text */
            text-align: center;  /* Center align footer text */
            padding: 20px 0;  /* Add padding for spacing */
            font-size: 14px;  /* Adjust font size */
        }

        /* Headings */
        h1, h2, h3 {
            color: #007BFF;  /* Blue headings */
            font-weight: bold;  /* Bold headings */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for additional options
st.sidebar.header("Travel Preferences")
st.sidebar.write("Select your preferences for a more tailored experience.")
destination = st.sidebar.selectbox("Choose a destination:", ["Riyadh", "Jeddah", "Dammam", "Medina", "Other"])
activity_type = st.sidebar.multiselect("Select activities:", ["Sightseeing", "Adventure", "Cultural", "Food", "Shopping"])

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Enter your question:")
if user_input:
    # Generate response
    with st.spinner("Generating response..."):
        response = rag.query(user_input)
    # Save to chat history
    st.session_state.chat_history.append({"user": user_input, "ai": response})

# Display chat history
st.subheader("Chat History")
for message in st.session_state.chat_history:
    st.markdown(f"**You:** {message['user']}")
    st.markdown(f"**AI:** {message['ai']}")
    st.markdown("---")  # Add a horizontal line for separation

# Add a footer
st.markdown("<footer style='text-align: center;'>Made with ❤️ by Fatima & Taif </footer>", unsafe_allow_html=True)


