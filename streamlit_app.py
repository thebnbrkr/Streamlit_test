import streamlit as st
from streamlit_chat import message
import requests

# Streamlit UI with Chat
st.title('Just Boeing Things')

HF_TOKEN = "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"

from langchain.embeddings import HuggingFaceBgeEmbeddings
import qdrant_client

from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

client = qdrant_client.QdrantClient(
    "https://172bf761-4b1c-4b06-a99f-cec74ea238fc.us-east4-0.gcp.cloud.qdrant.io", 
    api_key="QJ7Mnibhy2rPAHPDJMHsx5VjgFBXIn264TxT_e4t7zJwnWHj3rJv3A",
)
collection_name = "Finance_Test"
qdrant = Qdrant(client, collection_name, model_norm)

query = "What's the interest rate on Ford's long term debt?"
found_docs = qdrant.similarity_search(query)
print(found_docs[0].page_content)


# Managing chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display messages with unique keys
for i, chat in enumerate(st.session_state.chat_history):
    message(chat["message"], is_user=chat["is_user"], key=str(i))

user_input = st.text_input("Enter your text here", key="input")

if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append({"message": user_input, "is_user": True})
        response = found_docs[0].page_content
        st.session_state.chat_history.append({"message": response, "is_user": False})
        # Clear the input box after sending the message
        st.experimental_rerun()
