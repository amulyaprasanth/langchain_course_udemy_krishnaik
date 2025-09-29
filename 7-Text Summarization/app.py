import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq




# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config("🦜 LangChain: Summarize Text from YT or website")
st.title("🦜 Langchain: Summarize Text from YT or website")

api_key = st.sidebar.text_input("Enter your Groq API key: ", type="password")
generic_url = st.text_input("Enter web page URL or YouTube video URL")
generate_summary = st.button("Generate Summary")

# Prompts
map_prompt = PromptTemplate.from_template(
    """Please summarize the below text:

    {text}

    Summary:"""
)

combine_prompt = PromptTemplate.from_template(
    """Provide a final summary of the given text.
    Add a title, introduction, and precise summary.
    Use bullet points for key highlights.

    {text}

    Summary:"""
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)

if api_key:  # ✅ only init LLM if key is given
    llm = ChatGroq(model="gemma2-9b-it", api_key=api_key)

    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True
    )

    if generate_summary:
        try:
            with st.spinner("Loading and summarizing..."):
                # Load documents
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, language="en")
                    docs = loader.load()
                else:
                    loader = WebBaseLoader(generic_url)
                    docs = loader.load()

                if not docs:
                    st.error("No content found to summarize.")
                else:
                    split_docs = text_splitter.split_documents(docs)
                    response = summarize_chain.invoke({"input_documents": split_docs})
                    st.success(response["output_text"])

        except Exception as e:
            st.error(f"Failed to generate summary: {e}")
else:
    st.warning("Please provide Groq API key to start summarizing.")
