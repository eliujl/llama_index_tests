#!pip3 install gradio
import gradio as gr
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex


def ingest():
    documents = SimpleDirectoryReader('./docs').load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    return index


def query(index, query_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response


def get_answer(query_text):
    response = query(index, query_text)
    return response


index = ingest()

iface = gr.Interface(
    fn=get_answer,
    inputs=[gr.inputs.Textbox(label="Enter your query here")],
    outputs=[gr.outputs.Textbox(label="Results")],
    capture_session=True,
    title="SimpleQA"
)

iface.launch()
