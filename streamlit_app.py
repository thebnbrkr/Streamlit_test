import streamlit as st
from streamlit_chat import message
import requests

#### Bunch of shit

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.retrievers import BM25Retriever, EnsembleRetriever

#####

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



### Formatting


import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

### Adding mixtral this time

from langchain.llms import DeepInfra

api_token = st.secrets["DEEPINFRA_API_TOKEN"]

llm = DeepInfra(model_id="meta-llama/Llama-2-70b-chat-hf")
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "max_new_tokens": 1500,
    "top_p": 0.9,
}


qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=qdrant.as_retriever(search_kwargs={"k": 2}),
                                  return_source_documents=True)

llm_response = qa_chain(query)
process_llm_response(llm_response)

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
        st.session_state.chat_history.append({"message": process_llm_response(llm_response), "is_user": False})
        # Clear the input box after sending the message
        st.experimental_rerun()
