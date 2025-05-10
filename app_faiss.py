# app.py

import os
import glob
import streamlit as st


def load_env_file(file_path):
    with open(file_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Load your .env file
load_env_file('.env')

# Now you can access your environment variables using os.environ
your_variable = os.getenv('YOUR_ENV_VARIABLE')


# LangChain imports
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")

MODEL = "gpt-4o-mini"

st.title("🧠 RAG-based Chat Assistant")

# Load and split documents
@st.cache_resource
def load_data():
    folders = glob.glob("knowledge-base/**/*", recursive=True)
    loaders = [TextLoader(path) for path in folders if path.endswith(".txt")]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return split_docs

# Initialize FAISS vector store
@st.cache_resource
def create_vector_store(docs):
    embeddings = OpenAIEmbeddings(model=MODEL)
    db = FAISS.from_documents(docs, embeddings)
    return db

docs = load_data()
db = create_vector_store(docs)

# Set up the conversational chain
llm = ChatOpenAI(model=MODEL)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=db.as_retriever(), memory=memory
)

# Chat Interface
user_input = st.chat_input("Ask me anything about your documents:")
if user_input:
    result = qa_chain.invoke({"question": user_input})
    st.write(result['answer'])
