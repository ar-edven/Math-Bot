import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from PIL import Image
import pytesseract

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Math Bot", page_icon="🤖")

st.title("Math Bot")
# Function to process the image and extract text using OCR
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to get response from AI model
def get_response(query, chat_history):
    template = """
    you are a helpful assistant. Keep that in mind use Latex format and also add $$ at the beginning and end of your response.
    
    Answer the following questions considering 
    
    Chat history: {chat_history}

    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI()

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "chat_history": chat_history,
        "user_question": query
    })



# Conversation history display
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User input section
user_query = st.chat_input("Your Message")
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process image and extract text if an image is uploaded
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    extracted_text = extract_text_from_image(image)
    st.markdown(f"Extracted Text: {extracted_text}")
    user_query = user_query + " " + extracted_text if user_query else extracted_text

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        r = fr'''{ai_response}'''
        st.markdown(r)
        print(r)
    st.session_state.chat_history.append(AIMessage(ai_response))

