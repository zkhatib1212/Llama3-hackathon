from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from dotenv import load_dotenv
import os
import nest_asyncio

nest_asyncio.apply()

parser = LlamaParse(
    result_type="text",
    api_key=os.getenv("SECRET_API_KEY")
)

file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(
    "./data",
    recursive=True,
    file_extractor=file_extractor,
).load_data()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = Ollama(model="llama3", request_timeout=360.0)

index = VectorStoreIndex.from_documents(documents,)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions."
        "You know a lot about cars and are willing to help people with"
        "diagnostic checks before they bring their car to a mechanic."
    ),
)
