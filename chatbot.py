import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Config
DATA_PATH = "data"
FAISS_PATH = "faiss_index"

# Initialize embeddings and LLM
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

# --- Document ingestion (run once to build FAISS index) ---
# Example: load a PDF
loader = PyPDFLoader("data/NIPS-2017-attention-is-all-you-need-Paper.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Build, saving & loading FAISS index
vector_store = FAISS.from_documents(chunks, embeddings_model)
vector_store.save_local(FAISS_PATH)
vector_store = FAISS.load_local(FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# --- Streamlit UI setup ---
st.set_page_config(page_title="LangChain RAG Chatbot", layout="wide")
st.title("💬 LangChain RAG Chatbot") 

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display existing chat history
for user_msg, bot_msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

# Chat input
user_input = st.chat_input("Ask me something...")

if user_input:
    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant docs
    docs = retriever.invoke(user_input)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    # Build RAG prompt
    rag_prompt = f"""
    You are an assistant that answers questions based only on the provided knowledge.
    Do not use internal knowledge.

    Question: {user_input}

    Conversation history: {st.session_state.history}

    Knowledge: {knowledge}
    """
    # Stream response
    response_text = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for response in llm.stream(rag_prompt):
            response_text += response.content
            message_placeholder.markdown(response_text)

    # Append AFTER streaming
    st.session_state.history.append((user_input, response_text))