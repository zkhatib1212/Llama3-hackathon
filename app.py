import nest_asyncio
nest_asyncio.apply()
import os
from glob import glob
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core.memory import ChatMemoryBuffer
import streamlit as st


parser = LlamaParse(result_type='text',api_key=os.getenv('LLAMA_CLOUD_KEY'))

file_extractor = {".pdf" : parser}

documents = SimpleDirectoryReader(input_files= glob('data/*'),file_extractor=file_extractor).load_data()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=360)
index = VectorStoreIndex.from_documents(documents,)
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
chat_engine = index.as_chat_engine(chat_mode="context",memory=memory, system_prompt=("You are a chatbot. able to have normal interactions"
                                                                                     "You know a lot about cars and willing to help people with"
                                                                                    "car issues by diagnosing the problems, expecially the data you've been trained on"),)


# Streamlit UI
st.title("MECH-AI-NIC")
st.write("You can ask me anything about cars! Im your Car Doctor!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def update_chat_history(user_message, bot_response):
    st.session_state.chat_history.append(f"You: {user_message}")
    st.session_state.chat_history.append(f"Bot: {bot_response}")

user_input = st.text_input("You:", key="user_input")
if st.button("Send"):
    if user_input:
        try:
            response = chat_engine.chat(user_input)
            update_chat_history(user_input, response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please enter a message.")

for chat in st.session_state.chat_history:
    st.write(chat)
