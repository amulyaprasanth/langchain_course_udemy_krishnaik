import math
import numexpr
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config("Solve Math Problems and Wiki Search Agent", page_icon="🦜")
st.title("🦜 Solve Math Problems + Wiki Search Agent")


# -------------------------------
# Sidebar: API Key Input
# -------------------------------
api_key = st.sidebar.text_input("Enter your Groq API key", type="password")


# -------------------------------
# Prompt for ReAct Agent
# -------------------------------
template = """Answer the following questions as best you can. 
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)


# -------------------------------
# Tools Setup
# -------------------------------
# Wikipedia tool
wiki_api_wrapper = WikipediaAPIWrapper(doc_content_chars_max=500, top_k_results=1)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

# Calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library."""
    local_dict = {"pi": math.pi, "e": math.e}
    expression = expression.replace("^", "**")
    return str(numexpr.evaluate(expression.strip(), local_dict=local_dict))

tools = [wiki_tool, calculator]


# -------------------------------
# Chat History
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi 👋, I am your assistant! How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


# -------------------------------
# User Input
# -------------------------------
if user_question := st.chat_input("Enter your question here..."):
    st.chat_message("user").write(user_question)
    st.session_state["messages"].append({"role": "user", "content": user_question})

    if not api_key:
        st.warning("⚠️ Please provide a valid Groq API key in the sidebar.")
    else:
        llm = ChatGroq(model="gemma2-9b-it", api_key=api_key)

        # Create agent
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

        # Streamlit callback handler
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

        # Create agent executor with intermediate steps
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
        )

        with st.spinner("Thinking..."):
            response = agent_executor.invoke(
                {"input": user_question},
                {"callbacks": [st_cb]}, 
            )

        # Get only final output text
        final_answer = response.get("output", str(response))

        # Display assistant response
        st.chat_message("assistant").write(final_answer)

        # Save to history
        st.session_state["messages"].append({"role": "assistant", "content": final_answer})

