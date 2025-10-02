import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

## Load the NVIDIA api key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("11-Nvidia NIM/us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


    
st.title("Nvidia NIM Demo")
st.set_page_config("🦜Nvidia NIM demo")


prompt = ChatPromptTemplate.from_template(
    """

    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context} 
    </context>

    questions: {input}
    """
)

if st.button("Document Embedding"):
    vector_embedding()
    st.write("FAISS vector store do is ready using NVIDIAEmbeddings")

prompt1 = st.text_input("Enter your Question from Documents")

if prompt1:
    documents_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    chain = create_retrieval_chain(retriever, documents_chain)

    start = time.process_time()
    response = chain.invoke({"input": prompt1})
    print("Response time: ", time.process_time() - start)
    st.write(response['answer'])


    # with a streamlit expander
    with st.expander("Document Similarity Search"):
        # find the relavant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-------------------------------------------------')
