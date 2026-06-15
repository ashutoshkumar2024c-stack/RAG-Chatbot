from dotenv import load_dotenv
import os

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Document loading
from langchain_community.document_loaders import PyPDFLoader

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# LLM
from langchain_mistralai import ChatMistralAI

# Chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load PDF
loader = PyPDFLoader(r"C:\Users\ashuk\Desktop\SEM-4\Embedded\module 2.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# Embeddings (local, no API needed)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Vector store
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Mistral LLM
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
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False
)

print("\n✅ RAG Chatbot ready! Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    response = chain.invoke({"question": question})
    print(f"\nAI: {response['answer']}\n")
