import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    """
    Loads the environment variables from the .env file

    Args:
    ---------------
        None.
    
    Returns:
    --------------
        None.
    """
    _ = load_dotenv(find_dotenv())

def get_openai_api_key() -> str:
    """
    Gets the OpenAI API key from the environment variables.

    Args:
    ---------------
        None.
    
    Returns:
    ---------------
        str: An OpenAI API key.
    """ 
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key