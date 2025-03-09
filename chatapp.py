import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit page config with custom title & layout
st.set_page_config(page_title="📚 AskMyPDF", page_icon="🤖", layout="wide")

# Apply custom CSS for cool design
st.markdown("""
    <style>
    /* Change background and text */
    body, [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: white;
    }
    
    /* Style chat messages */
    .chat-container {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-size: 18px;
    }
    
    .bot-message {
        background-color: #1E1E1E;
        border-left: 5px solid #00C853;
    }

    .user-message {
        background-color: #263238;
        border-left: 5px solid #1E88E5;
    }

    /* Style buttons */
    .stButton>button {
        background-color: #6200EA !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
    }

    /* Custom text input */
    .stTextInput>div>div>input {
        background-color: #263238 !important;
        color: white !important;
        border: 1px solid #00C853 !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks for better processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Store vector embeddings using FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# AI Model & Chat Chain
def get_conversational_chain():
    prompt_template = """
    Answer the question with details from the provided context. If the answer is not in the context, reply:
    "❌ Answer is not available in the provided documents."
    
    **Context:** 
    {context}
    
    **User Question:** 
    {question}

    **Response:**
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# User query processing
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Styled chatbot reply
    st.markdown(f"""
    <div class="chat-container bot-message">
        <strong>🤖 AI Reply:</strong> <br>{response["output_text"]}
    </div>
    """, unsafe_allow_html=True)

# Sidebar for file uploads
with st.sidebar:
    st.image("img/Robot.jpg", use_container_width=True)
    st.title("📁 Upload PDFs")
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

    if st.button("Process Documents"):
        with st.spinner("Processing PDFs... ⏳"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("✅ Documents processed!")

    st.write("---")
    st.image("img/gkj.jpg", use_container_width=True)
    st.write("🚀 AI Chat App by **Suyash Kusumkar**")

# Main UI
st.title("📚 AskMyPdf - > Upload Multi PDF 🤖")
st.write("Ask questions about your uploaded PDFs, and the AI will provide insights!")

user_question = st.text_input("💬 Type your question here...")
if user_question:
    st.markdown(f"""
    <div class="chat-container user-message">
        <strong>🧑‍💻 You:</strong> <br>{user_question}
    </div>
    """, unsafe_allow_html=True)
    user_input(user_question)

# Custom Footer
st.markdown("""
    <div style="text-align: center; padding: 10px; background-color: #0E1117;">
        <a href="https://github.com/yashkusumkar02" target="_blank" style="color: #1E88E5; text-decoration: none;">
        🚀 Created by Suyash Kusumkar | Made with ❤️
        </a>
    </div>
""", unsafe_allow_html=True)

def main():
    st.title("📚 Multi-PDF Chatbot 🤖")
    st.write("Ask questions about your uploaded PDFs, and the AI will provide insights!")

    user_question = st.text_input("💬 Type your question here...")
    
    if user_question:
        st.markdown(f"""
        <div class="chat-container user-message">
            <strong>🧑‍💻 You:</strong> <br>{user_question}
        </div>
        """, unsafe_allow_html=True)
        user_input(user_question)

