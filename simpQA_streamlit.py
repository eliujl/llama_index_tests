import streamlit as st
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
import os
import shutil


def save_file(files):
    directory_name = 'tmp_docs'

    # Remove existing files in the directory
    if os.path.exists(directory_name):
        for filename in os.listdir(directory_name):
            file_path = os.path.join(directory_name, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error: {e}")

    # Save the new file with original filename
    if files is not None:
        for file in files:
            file_name = file.name
            file_path = os.path.join(directory_name, file_name)
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file, f)


def ingest(docs_dir):
    documents = SimpleDirectoryReader(docs_dir).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    return index


def get_answer(index, message):
    response = query(index, message)
    return [('Chatbot', ''.join(response.response))]


def query(index, query_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response


# Initialize chatbot history
chatbot = []

# Display file upload component
files = st.file_uploader('Upload Files', accept_multiple_files=True)
if files is not None:
    save_file(files)


index = ingest('tmp_docs')

# Display message input component
message = st.text_input('Enter message')

# If message is entered, ingest documents and get chatbot response
if message:
    chatbot.append(('You', message))
    chatbot += get_answer(index, message)

# Display chat history
st.text_area('Chatbot:', value='\n'.join(
    [f'{x[0]}: {x[1]}' for x in chatbot]), height=250)
