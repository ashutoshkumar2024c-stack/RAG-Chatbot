import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or st.secrets.get("MISTRAL_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Chatbot")
st.caption("Ask questions from your own documents!")

# Initialize session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🚀 Process Document"):
            with st.spinner("Processing your document..."):

                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Load and split
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(documents)

                # Embeddings and vector store
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )

                # LLM
                llm = ChatMistralAI(
                    model="mistral-tiny",
                    temperature=0.7,
                    mistral_api_key=MISTRAL_API_KEY
                )

                # Memory
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )

                # Chain
                st.session_state.chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    verbose=False
                )
                st.session_state.messages = []
                os.remove(temp_path)

            st.success(f"✅ Ready! {len(chunks)} chunks processed.")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Click Process")
    st.markdown("3. Ask questions!")

# Chat interface
if st.session_state.chain is None:
    st.info("👈 Upload a PDF from the sidebar to get started!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke(
                    {"question": prompt}
                )
                answer = response["answer"]
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )