import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    streaming=True
)

def stream_response(message, history):
    print(f"DEBUG: Prompt: {message}. History: {history}\n")

    history_langchain_format = []
    # Add a system message if needed.
    history_langchain_format.append(SystemMessage(content="You are a helpful assistant."))

    # Process each message from the history (now in dict format)
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            history_langchain_format.append(HumanMessage(content=content))
        elif role == "assistant":
            history_langchain_format.append(AIMessage(content=content))
        elif role == "system":
            history_langchain_format.append(SystemMessage(content=content))

    # Add the new human message if provided
    if message is not None:
        history_langchain_format.append(HumanMessage(content=message))
        partial_message = ""
        for response in llm.stream(history_langchain_format):
            partial_message += response.content
            yield partial_message

# Define the UI (chat interface)
ui = gr.ChatInterface(
    stream_response,
    type="messages",
    textbox=gr.Textbox(
        placeholder="Ask a question ...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

# Start the UI
ui.launch(share=True, debug=True)