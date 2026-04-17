import os
from utils import get_openai_api_key
from openai import OpenAI

openai_api_key = get_openai_api_key()

client = OpenAI(
  api_key=openai_api_key
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", 
     "content": "What is the weather like in Cape Town, South Africa?"
    }
  ]
)

print(completion.choices[0].message)
