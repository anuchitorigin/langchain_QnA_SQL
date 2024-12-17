from langchain_community.utilities import SQLDatabase
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import QuerySQLDatabaseTool


### Constants ###
SQL_PREFIX = """
    You are an agent designed to interact with a SQL database especially for SQLite.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    Pay attention to use date('now') function to get the current date, if the question involves "today".
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables.
"""
# QUESTION = "How many employees are there"
QUESTION = "What is the name of the artist with ID = 10" #"How many albums does alis in chain have?"

### Connect to SQL database ###
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
# print(db.run("SELECT * FROM Artist LIMIT 10;"))


### Connect to OpenAI model ##
# llm = ChatOpenAI(model="gpt-4o-mini", api_key="...")
llm = ChatOllama(model="llama3.2")


### Convert question to SQL query ###
# chain = create_sql_query_chain(llm, db)
# chain.get_prompts()[0].pretty_print()

# response = chain.invoke({"question": QUESTION})
# print(response)
# print(db.run(response))


### Execute SQL query ###
write_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDatabaseTool(db=db)
# chain = write_query | execute_query
# print(chain.invoke({"question": QUESTION})) #


### Answer the question ###
answer_prompt = PromptTemplate.from_template("""
    Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: 
""")

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke({"question": QUESTION}))