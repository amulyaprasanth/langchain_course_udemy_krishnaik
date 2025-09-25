import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()


## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant, please respond to the user queries"),
    ("user", "Question: {question}")]
)

# Create a function generate response
def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.apikey = api_key
    llm = ChatOpenAI(model = llm, temperature=temperature, max_tokens =max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    answer = chain.invoke({"question": question})

    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot with OpenAI")

## sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter you Open AI API Key: ", type="password")

# Drop down to select variaous llm model from openai
llm = st.sidebar.selectbox("Open AI Model: ", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value = 0.0, max_value = 1.0, value = 0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value = 50, max_value = 300, value = 150)


## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input and api_key:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)

    with st.spinner("Thinking..."):
         st.write(response)
