import streamlit as st
from streamlit_chat import message
import requests

#### Bunch of shit

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

"""
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
"""

from langchain.embeddings import HuggingFaceInstructEmbeddings

#####

# Streamlit UI with Chat
st.title('מותק ❤️')

HF_TOKEN = "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"

from langchain.embeddings import HuggingFaceBgeEmbeddings
import qdrant_client

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



## Trying Mixtral

from langchain.llms import DeepInfra

api_token = st.secrets["DEEPINFRA_API_TOKEN"]

llm = DeepInfra(model_id="mistralai/Mixtral-8x7B-Instruct-v0.1")
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "max_new_tokens": 1500,
    "top_p": 0.9,
}


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
        review_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=qdrant.as_retriever())
        st.session_state.chat_history.append({"message": review_chain.run(user_input), "is_user": False})
        # Clear the input box after sending the message
        st.experimental_rerun()
