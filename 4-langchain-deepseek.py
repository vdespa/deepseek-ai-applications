
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# Load environment variables from .env file
load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    streaming=True
)

messages = [
    ("system", "You are a helpful assistant. Answer as brief as possible."),
    ("human", "In which US state is the city of Ryolithe?"),
]
response = llm.invoke(messages)

print(response.content)