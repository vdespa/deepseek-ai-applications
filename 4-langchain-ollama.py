from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="deepseek-r1:1.5b")

messages = [
    ("system", "You are a helpful assistant. Answer as brief as possible."),
    ("human", "In which US state is the city of Ryolithe?"),
]
response = llm.invoke(messages)

print(response)