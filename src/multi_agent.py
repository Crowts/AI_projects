import os
from utils.utils import get_openai_api_key
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

hosted_model = ChatOpenAI(model="gpt-4o-mini")

local_llm = "qwen2.5:7b"
Model = ChatOllama(model=local_llm, temperature=0.0)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.environ["SERPER_API_KEY"]
LANGSMITH_TRACING = os.environ["LANGSMITH_TRACING"]
LANGSMITH_ENDPOINT = os.environ["LANGSMITH_ENDPOINT"]
LANGSMITH_API_KEY = os.environ["LANGSMITH_API_KEY"]
LANGSMITH_PROJECT = os.environ["LANGSMITH_PROJECT"]


def add(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
    ---------------
        a: A number of type float.
        b: A number of type float.

    Return:
    ---------------
        The sum of a and b.
    """
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
    ---------------
        a: A number of type float.
        b: A number of type float.

    Returns:
    ---------------
        The product of a and b.
    
    
    """
    return a * b

def web_search(query: str) -> str:
    """Searches the web for information given a query using Google Serper API Wrapper.

    Args:
    ---------------
        query: A string

    Returns:
    ---------------
        Text output in string format.
    """
    search = GoogleSerperAPIWrapper() # uses os.environ["SERPER_API_KEY"]
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error during web search: {e}"

math_agent = create_react_agent(
    model=hosted_model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=hosted_model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=hosted_model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})

for m in result["messages"]:
    m.pretty_print()