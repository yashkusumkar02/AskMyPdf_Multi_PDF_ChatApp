import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Using free API key

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
    
    /* Table styling */
    .dataframe {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    
    .dataframe th {
        background-color: #6200EA !important;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #263238 !important;
    }
    
    .dataframe tr:nth-child(odd) {
        background-color: #1E1E1E !important;
    }
    
    </style>
""", unsafe_allow_html=True)

def load_image(image_path):
    """Handle image loading with proper path checking"""
    try:
        if Path(image_path).exists():
            return image_path
        return "https://via.placeholder.com/150"
    except:
        return "https://via.placeholder.com/150"

def clean_table_data(table):
    """Clean and process extracted table data with proper column handling"""
    if not table or len(table) == 0:
        return None
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(table)
        
        # Handle empty DataFrames
        if df.empty:
            return None
        
        # Clean column headers
        cols = []
        count = {}
        for idx, col in enumerate(df.columns):
            col_name = str(col) if (col is not None and str(col).strip() != "") else f"Column_{idx}"
            
            # Handle duplicates
            if col_name in count:
                count[col_name] += 1
                cols.append(f"{col_name}_{count[col_name]}")
            else:
                count[col_name] = 0
                cols.append(col_name)
        
        df.columns = cols
        
        # Clean empty rows and columns
        df = df.replace(['', ' ', None, np.nan, 'None', 'NaN'], np.nan)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        return df.reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Error cleaning table: {str(e)}")
        return None

def extract_pdf_content(pdf_docs):
    text_content = ""
    tables_content = []
    
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    # Extract text
                    text = page.extract_text()
                    if text:
                        text_content += text + "\n\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        cleaned_table = clean_table_data(table)
                        if cleaned_table is not None and not cleaned_table.empty:
                            tables_content.append(cleaned_table)
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            continue
    
    return text_content, tables_content

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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
    
    # Using gemini-1.5-flash (free model)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        # Styled chatbot reply with source context
        st.markdown(f"""
        <div class="chat-container bot-message">
            <strong>🤖 AI Reply:</strong> <br>{response["output_text"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Show source context
        st.markdown("<div style='margin-top: 10px; font-size: 14px; color: #aaa;'>Source context:</div>", unsafe_allow_html=True)
        for i, doc in enumerate(docs[:2]):  # Show top 2 relevant sources
            st.text_area(f"Source {i+1}", value=doc.page_content[:500] + "...", height=100, label_visibility="collapsed")
            
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

# Sidebar for file uploads
with st.sidebar:
    # Use absolute path for images
    robot_img = load_image(os.path.join("img", "Robot.jpg"))
    profile_img = load_image(os.path.join("img", "gkj.png"))
    
    st.image(robot_img, use_container_width=True)
    st.title("📁 Upload PDFs")
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type="pdf")

    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing PDFs... ⏳"):
                # Extract content
                raw_text, tables = extract_pdf_content(pdf_docs)
                
                if not raw_text and not tables:
                    st.warning("No extractable content found in documents")
                    st.stop()
                
                # Process text for vector store
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                
                # Store extracted content
                st.session_state['extracted_text'] = raw_text
                st.session_state['extracted_tables'] = tables
                st.session_state['processed'] = True
                
                st.success("✅ Documents processed successfully!")
        else:
            st.warning("Please upload PDF files first!")

    st.write("---")
    st.image(profile_img, use_container_width=True)
    st.write("🚀 AI Chat App by **Suyash Kusumkar**")

# Main UI
st.title("📚 AskMyPDF - Multi-PDF Chatbot 🤖")
st.write("Upload your PDFs, extract text and tables, and ask questions about the content!")

# Only show these sections if documents have been processed
if st.session_state.get('processed', False):
    # Display extracted content
    if 'extracted_text' in st.session_state or 'extracted_tables' in st.session_state:
        st.subheader("📄 Extracted Content from PDFs")
        
        # Display text content
        if 'extracted_text' in st.session_state and st.session_state.extracted_text:
            with st.expander("📝 View Extracted Text", expanded=False):
                st.text_area("Text Content", 
                            value=st.session_state.extracted_text, 
                            height=300,
                            label_visibility="collapsed")
        
        # Display tables
        if 'extracted_tables' in st.session_state and st.session_state.extracted_tables:
            with st.expander("📊 View Extracted Tables", expanded=False):
                for i, table in enumerate(st.session_state.extracted_tables):
                    st.write(f"### Table {i+1}")
                    try:
                        st.dataframe(table)
                    except Exception as e:
                        st.warning(f"Couldn't display table {i+1} properly: {str(e)}")
                        st.write(table.to_dict())  # Fallback display
                    st.write("---")

    # Chat interface - Only show if documents are processed
    st.subheader("Ask Questions About Your Documents")
    user_question = st.text_input(
        "💬 Type your question here...",
        placeholder="What information are you looking for?",
        label_visibility="collapsed"
    )

    if user_question:
        st.markdown(f"""
        <div class="chat-container user-message">
            <strong>🧑‍💻 You:</strong> <br>{user_question}
        </div>
        """, unsafe_allow_html=True)
        user_input(user_question)

# Custom Footer
st.markdown("""
    <div style="text-align: center; padding: 20px; margin-top: 50px; background-color: #0E1117; border-radius: 10px;">
        <a href="https://github.com/yashkusumkar02" target="_blank" style="color: #1E88E5; text-decoration: none;">
        🚀 Created by Suyash Kusumkar | Made with ❤️ using Streamlit & Gemini
        </a>
    </div>
""", unsafe_allow_html=True)