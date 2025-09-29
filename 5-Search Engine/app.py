import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# create arxiv api wrapper and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Create duckduck go run
search = DuckDuckGoSearchRun(name="Search")

st.title("Langchain Chat With Search")

st.markdown(""" In this example we are using `Streamlit callback handler` to display the thoughts and actions of the agent.  
Try more Langchain StreamLit examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).""")

# sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant",
         "content": "Hi, I'm a chat bot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if api_key:
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key,
                   model_name="gemma2-9b-it", streaming=True)

    # Define tools
    tools = [search, arxiv, wiki]

    # Load prompt from LangChain hub
    prompt = hub.pull("hwchase17/react")

    # Create a ReAct-style agent
    search_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    search_agent_executor = AgentExecutor(agent=search_agent, tools=tools, verbose=True)
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # add the user message to the chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent_executor.invoke(
            {"input": prompt},
            config={"callbacks": [st_cb]}
        )
        final_output = response["output"]

        # reuse the same container
        output_placeholder = st.empty()
        output_placeholder.markdown(final_output)

        st.session_state.messages.append({"role": "assistant", "content": final_output})

