from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
import os
import nest_asyncio

# Allow async operations
# Primarily for file parser
nest_asyncio.apply()

# Create Parser object
parser = LlamaParse(
    result_type="text",
    api_key=os.getenv("SECRET_API_KEY")
)
file_extractor = {".pdf": parser, ".csv": parser}

# Extract/Parse documents
documents = SimpleDirectoryReader(
    "./data",
    recursive=True,
    file_extractor=file_extractor,
).load_data()

# Define the index, model settings, etc. for the eventual chatbot object
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
index = VectorStoreIndex.from_documents(documents,)
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Context to provide to the chatbot
system_prompt = """
You are a chatbot, able to have normal interactions.
You know a lot about cars and are willing to help people with
diagnostic checks before they bring their car to a mechanic.

You already have knowledge of the owner's manual and other documentation
for certain car models, as well as an OBD-II code reference.

If someone provides an OBD-II code or a specific issue with their vehicle,
please provide a solution, approximate parts cost, and approximate
labour hours.
"""

# Create the chat engine object
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=system_prompt,
)

def ask_chatbot(text: str) -> None:
    """
    Pass a chat message through to the chatbot and print a response.

    Parameters
    ---------
    text (str):
        The message to pass along to the chatbot.
    """

    response = chat_engine.chat(text)
    print(response)