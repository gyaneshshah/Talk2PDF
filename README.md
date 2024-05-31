# Talk2PDF

Talk2PDF is an application that allows users to interact with their PDF files, enabling them to have a conversation or ask questions about the content.

## Features

- **Interactive Conversations**: Users can talk to their PDFs, asking questions and receiving detailed responses.
- **High Accuracy**: Implemented Retrieval Augmented Generation (RAG) for improved accuracy and efficiency.
- **Customizable**: Users can fine-tune the model by adjusting chunk size and chunk overlap.
- **Easy Deployment**: Deployed using Streamlit for a seamless user experience.

## Technologies Used

- **OpenAI API**:
  - **Language Model**: [`gpt-3.5-turbo`](https://openai.com/research/gpt-3-5)
  - **Embedding Model**: [`text-embedding-ada-002`](https://platform.openai.com/docs/guides/embeddings)
- **Pinecone**: Used for [vector database management](https://www.pinecone.io/).
- **Langchain**: Used for [building applications with LLMs](https://www.langchain.com/).
- **Streamlit**: Utilized for [deploying the application](https://streamlit.io/).

## Deployment

The project is deployed and can be accessed [here](https://talk-2-pdf.streamlit.app).

Feel free to explore and interact with your PDFs in a whole new way!
