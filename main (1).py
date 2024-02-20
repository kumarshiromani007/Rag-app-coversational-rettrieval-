import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from ctransformers import AutoConfig



st.title("RAG Aap")

# Load data from CSV file
file_path = "scraped_data.csv"  
loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

# Download Sentence Transformers Embedding From Hugging Face
embeddings2 = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
embeddings=HuggingFaceEmbeddings(model_name='hkunlp/instructor-large')

# Convert the text chunks into embeddings and save the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)

# Load LLM model

config = AutoConfig.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
config.config_max_new_tokens = 2000
config.config_context_length = 4000
config.temperature = 0.4 

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML",
                    model_type="llama",
                    max_new_tokens=config.config_max_new_tokens,
                    temperature=0.4,
                    )


# Create conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

# Input field for user query
query =  st.text_input("Input Prompt:")

if st.button("Get Response"):
    if query:
        result = qa({"question": query, "chat_history": []})
        st.write("Response:", result['answer'])
    else:
        st.warning("Please enter a query.")
