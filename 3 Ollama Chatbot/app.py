import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()


## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant, please respond to the user queries"),
    ("user", "Question: {question}")]
)

# Create a function generate response
def generate_response(question, engine, temperature, max_tokens):
    llm = OllamaLLM(model = engine, temperature=temperature, max_tokens =max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    answer = chain.invoke({"question": question})

    return answer

# Drop down to select variaous llm model from openai
engine = st.sidebar.selectbox("Open Source Modelsl: ", ["gemma:2b", "mistral:7b"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value = 0.0, max_value = 1.0, value = 0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value = 50, max_value = 300, value = 150)


## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)

    with st.spinner("Thinking..."):
         st.write(response)