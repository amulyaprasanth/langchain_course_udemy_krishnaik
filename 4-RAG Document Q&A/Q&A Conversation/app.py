# RAG Q&A Conversation with PDF Including Chat history
from io import BytesIO
import tempfile
from langchain_core.runnables import RunnableWithMessageHistory
import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

# load hugging face env token
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# setup streamlit app
st.title("Conversational RAG with PDF Uploads and chat history")

st.write("Upload pdf's and chat with their content")


# INput the Groq api key
api_key = st.text_input("Enter your Groq API key: ", type="password")

# check if groq api key is provided

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

    # Chat Interfact
    session_id = st.text_input("Session ID", value="default_session")

    # statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Choose a PDF file.", accept_multiple_files=True)

    # process uploaded PDF'S
    documents = []
    if not uploaded_files:
        st.write("please upload a pdf file to continue")

    else:
        for uploaded_file in uploaded_files:
            # Use BytesIO for in-memory buffer
            file_bytes = BytesIO(uploaded_file.getvalue())

            # Create a temporary file for PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes.read())
                tmp_file.flush()
                loader = PyPDFLoader(tmp_file.name)
                docs = loader.load()
                documents.extend(docs)

    # Split and create embeddigns for the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=500)

    if documents:
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
    else:
        retriever = None

    # new prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the last user question which"
        "which might reference context in the chat history."
        "formulate a standalone question which can be understood"
        "without the chat history. Do NOT answer the question,"
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # create history aware retriever
    if retriever:
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt)

    # Answer question
    system_prompt = (
        """
        You are an assistant for question-answering tasks.
        Use th following pieces of retreived context to answer
        the question. If you don't know the answer, say that you dont' know.
        Use three sentences maximum to keep the answer concise.
        \n
        \n
        {context} 
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    documents_chain = create_stuff_documents_chain(
        llm, qa_prompt, output_parser=StrOutputParser())
    qa_chain = create_retrieval_chain(
        history_aware_retriever, combine_docs_chain=documents_chain)

    # create a get session histor fucntion

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your Question")

    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable":
                    {"session_id": session_id}}
        )

        with st.expander("Store"):
            st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])

        with st.expander("Chat History"):
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")
