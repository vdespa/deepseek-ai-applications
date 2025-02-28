import os
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import gradio as gr

# This prevents the tokenizers from attempting parallel processing after a fork, which triggers a warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Extract text from all PDF files in the "files" directory
combined_text = ''
files_directory = 'files'
for filename in os.listdir(files_directory):
    if filename.endswith('.pdf'):
        with pdfplumber.open(os.path.join(files_directory, filename)) as pdf:
            for page in pdf.pages:
                combined_text += page.extract_text() + ' '

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_chunks = text_splitter.split_text(combined_text)

# Create embeddings and build a vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
db = FAISS.from_texts(text_chunks, embeddings)
retriever = db.as_retriever()

# Prepare a prompt template for the chain
template = """
    Answer the question based only on the following context:
    {context}

    Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

# Create a chain that combines documents retrieved from the vectorstore with the prompt
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Define a function that wraps the chain call
def answer_question(question):
    result = rag_chain.invoke({"input": question})
    return result.get("answer", "No answer found.")

# Test the chain by printing a sample answer
# print(answer_question("Who is the CEO of the company?"))

# Create and launch a Gradio interface
ui = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(placeholder="Ask a question about the PDFs...", label="Prompt"),
    outputs="text",
    title="Chat with your documents",
    description="Enter a question and receive an answer based on the PDF content.",
)

ui.launch(share=True, debug=True)