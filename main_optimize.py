import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Function to split the input text into smaller chunks
def split_text(text, max_length=512):
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        else:
            # Find the last space within the maximum length
            last_space_idx = text[:max_length].rfind(' ')
            if last_space_idx != -1:
                chunk = text[:last_space_idx]
                chunks.append(chunk)
                text = text[last_space_idx+1:]
            else:
                # If no space is found, split at the maximum length
                chunk = text[:max_length]
                chunks.append(chunk)
                text = text[max_length:]
    return chunks

#@st.cache
def load_data(file_path):
    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    return loader.load()


def split_text_into_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    return text_splitter.split_documents(data)


def load_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


def create_faiss_index(text_chunks, embeddings):
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    return docsearch


def load_llm_model():
    return CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", max_new_tokens=500, temperature=0.1)

st.title("Conversational Retrieval App")

file_path = "scraped_data.csv"
data = load_data(file_path)
text_chunks = split_text_into_chunks(data)
embeddings = load_embeddings()
docsearch = create_faiss_index(text_chunks, embeddings)
qa = ConversationalRetrievalChain.from_llm(load_llm_model(), retriever=docsearch.as_retriever())

query = st.text_input("Input Prompt:")

if st.button("Get Response"):
    if query:
        query_chunks = split_text(query)
        response_chunks = []
        for chunk in query_chunks:
            result = qa({"question": chunk, "chat_history": []})
            response_chunks.append(result['answer'])
        response = ' '.join(response_chunks)
        st.write("Response:", response)
    else:
        st.warning("Please enter a query.")
