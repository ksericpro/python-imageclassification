from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr
import os
import time
#openai.api_key = "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"

os.environ["OPENAI_API_KEY"] = "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"  # Replace with your key

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    #for i in range(len(gpt_response.content)):
    #    time.sleep(0.3)
    #    yield "You typed: " + message[: i+1]
    return gpt_response.content

gr.ChatInterface(predict).launch()