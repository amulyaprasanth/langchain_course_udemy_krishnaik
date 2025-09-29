import streamlit as st
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_cohere.sql_agent.agent import create_sql_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import AgentType
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
from urllib.parse import quote  # for parsing passwords with @

st.set_page_config(page_title="Langchain: Chat With SQL DB", page_icon="🐦")
st.title("🐦 LangChain: Chat With SQL DB")

INJECTION_WARNING = """ 
SQL agent can be vulnerabel to prompt injection. Use a DB role with limited permissions.
Read more [here](https://python.langchain.com/docs/security)
"""

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["User SQLite 3 Database - Student.db",
             "Connect to your SQL Database"]

selected_opt = st.sidebar.radio(label="Choose the DB which you want to chat",
                                options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    my_sql_host = st.sidebar.text_input("Provide My SQL Host ")
    my_sql_user = st.sidebar.text_input("MySQL User")
    my_sql_pass = st.sidebar.text_input("MySQL password", type="password")
    my_sql_db = st.sidebar.text_input("MySQL database")

else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label="Groq API key: ", type="password")

if not db_uri:
    st.info("Please enter the Database Information and uri")

if not api_key:
    st.info("Please add the groq api key")


@st.cache_resource(ttl="2h")
def configure_db(db_uri, my_sql_host=None, my_sql_user=None, my_sql_pass=None, my_sql_db=None):
    if db_uri == LOCALDB:
        # setup the local file path for the local db
        db_filepath = (Path(__file__).parent.parent/"student.db").absolute()

        def creator(): return sqlite3.connect(
            f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))

    elif db_uri == MYSQL:
        if not (my_sql_host and my_sql_user and my_sql_pass and my_sql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()

        return SQLDatabase(create_engine(f"""mysql+mysqlconnector://{my_sql_user}:{quote(my_sql_pass)}@{my_sql_host}/{my_sql_db}"""))


if db_uri == MYSQL:
    db = configure_db(db_uri, my_sql_host, my_sql_user, my_sql_pass, my_sql_db)

else:
    db = configure_db(db_uri)


# toolkit
llm = ChatGroq(api_key=api_key, model_name="gemma2-9b-it", streaming=True)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
