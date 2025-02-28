from ollama import chat
from ollama import ChatResponse
import re

response: ChatResponse = chat(
    model='deepseek-r1:1.5b', 
    messages=[
      {'role': 'user', 'content': 'In which US state is the city of Ryolithe?'}
])

print(response.message.content)

#cleaned_content = re.sub(r"<think>.*?</think>\n?", "", response.message.content, flags=re.DOTALL)
#print(cleaned_content)