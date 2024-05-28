import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os

# Function to load the pdf file
def load_pdf(file):
    from langchain.document_loaders import PyPDFLoader
    print(f'Loading {file}')
    loader = PyPDFLoader(file)
    data = loader.load()
    return data

# Function to create chunks of data
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create Embeddings
def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    from pinecone import ServerlessSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists!')
        print('Loading embeddings')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Embeddings successfully loaded!')
    else:
        print(f'Creating index {index_name} and embeddings...')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Embeddings successfully loaded!')
    return vector_store

# Get Answers
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.run(q)
    return answer