from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_ollama import (ChatOllama, OllamaEmbeddings)

from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings

from langgraph.prebuilt import create_react_agent

import ast
import re


def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


### Constants ###
SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""


### Connect to SQL database ###
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())


### Connect to OpenAI model ##
llm = ChatOllama(model="llama3.2", format="json")


### Create toolkit ###
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()
# print(tools)

system_message = SystemMessage(content=SQL_PREFIX)
agent = create_react_agent(llm, tools, state_modifier=system_message)

for s in agent.stream(
    {"messages": [HumanMessage(content="Describe the playlisttrack table")]}
):
    print(s)
    print("----")

# for s in agent.stream(
#     {"messages": [HumanMessage(content="Which country's customers spent the most?")]}
# ):
#     print(s)
#     print("----")


### Using Vector DB ###
# artists = query_as_list(db, "SELECT Name FROM Artist")
# albums = query_as_list(db, "SELECT Title FROM Album")
# # print(albums[:5])

# vector_db = FAISS.from_texts(artists + albums, OllamaEmbeddings(model="llama3.1"))
# retriever = vector_db.as_retriever(search_kwargs={"k": 5})
# description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
# valid proper nouns. Use the noun most similar to the search."""
# retriever_tool = create_retriever_tool(
#     retriever,
#     name="search_proper_nouns",
#     description=description,
# )
# print(retriever_tool.invoke("Alice Chains"))

# system = """You are an agent designed to interact with a SQL database.
# Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
# Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
# You can order the results by a relevant column to return the most interesting examples in the database.
# Never query for all the columns from a specific table, only ask for the relevant columns given the question.
# You have access to tools for interacting with the database.
# Only use the given tools. Only use the information returned by the tools to construct your final answer.
# You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

# DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

# You have access to the following tables: {table_names}

# If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
# Do not try to guess at the proper name - use this function to find similar ones.""".format(
#     table_names=db.get_usable_table_names()
# )

# system_message = SystemMessage(content=system)

# tools.append(retriever_tool)

# agent = create_react_agent(llm, tools, state_modifier=system_message)

# for s in agent.stream(
#     {"messages": [HumanMessage(content="How many albums does alis in chain have?")]}
# ):
#     print(s)
#     print("----")
