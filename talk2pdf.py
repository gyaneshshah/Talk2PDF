import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
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
def create_embeddings(chunks, index_name='talk2pdf'):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    from pinecone import ServerlessSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = Pinecone.from_existing_index(index_name, embeddings)
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

# Calculate Embedding Cost
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content))for page in texts])
    return total_tokens, total_tokens/1000*0.0004

# Clearing History
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('talk2PDF_logo.png')
    st.subheader('Have a conversation with your document!')

    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        uploaded_file = st.file_uploader('Upload a PDF File:', type=['pdf'])
        chunk_size = st.number_input('Chunk Size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('Chunk Overlap:', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Upload PDF', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading your PDF....'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                
                data = load_pdf(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk Size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding Cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks=chunks)

                st.session_state.vs = vector_store
                st.success('PDF Uploaded and Embedded Successfully!')

    q = st.text_input('Ask me a question about your PDF')
    if q:
        standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
                          "If you can't answer then return `I don't know, please try again after sometime`."
        q = f"{q} {standard_answer}"
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('Answer: ', value=answer)

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-"*100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)