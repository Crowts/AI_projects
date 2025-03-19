import os
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from rich.traceback import install 
install()

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

_ = load_dotenv(find_dotenv()) 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.environ["SERPER_API_KEY"]
LANGSMITH_TRACING = os.environ["LANGSMITH_TRACING"]
LANGSMITH_ENDPOINT = os.environ["LANGSMITH_ENDPOINT"]
LANGSMITH_API_KEY = os.environ["LANGSMITH_API_KEY"]
LANGSMITH_PROJECT = os.environ["LANGSMITH_PROJECT"]

local_llm = "llama3.2" #llama3.2 - qwen2.5:7b
Model = ChatOllama(model=local_llm, temperature=0.0)
remote_model = ChatOpenAI(model="gpt-4o-mini")

def add(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
    ---------------
        a: A float or integer value.
        b: A float or integer value.

    Returns:
    ---------------
        The sum of a and b.
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        logging.error(f"TypeError: Invalid input types - a: {type(a)}, b: {type(b)}")
        raise TypeError(f"Invalid input: a={a}, b={b}. Both must be int or float.")

    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
    ---------------
        a: A float or integer value.
        b: A float or integer value.

    Returns:
    ---------------
        The product of a and b.
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        logging.error(f"TypeError: Invalid input types - a: {type(a)}, b: {type(b)}")
        raise TypeError(f"Invalid input: a={a}, b={b}. Both must be int or float.")

    return a * b

def web_search(query: str) -> str:
    """Searches the web for information given a query using Google Serper API Wrapper.

    Args:
    ---------------
        query: A string.

    Returns:
    ---------------
        Text output in string format.
    """
    search = GoogleSerperAPIWrapper()
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error during web search: {e}"

math_agent = create_react_agent(
    model=Model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=Model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

workflow = create_supervisor(
    [research_agent, math_agent],
    model=Model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})

for m in result["messages"]:
    m.pretty_print()