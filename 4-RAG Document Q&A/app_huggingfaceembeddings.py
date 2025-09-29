import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load the groq api key
os.environ["OPENAI_API_KEY"] =  os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

## If you do not have open AI key use the below Huggingface embedding
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response on the queston
    <context>
    {context}
    <context>
    Question:{input}
    """)


def create_vector_embedding():
    if  "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader("4 RAG Document Q&A/research_papers")
        st.session_state.docs  = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A with Groq and Gemma2")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print(f"Response time: {time.process_time()-start}")

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity search"):
        st.write(response['context'])