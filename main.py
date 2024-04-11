import asyncio
# Initialize asyncio loop and set environment variables
asyncio.new_event_loop()
asyncio.set_event_loop(asyncio.new_event_loop())

import os
import pandas as pd
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI

llm = OpenAI(temperature=0.1, model="gpt-4")


os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Load and prepare data
df = pd.read_csv("chat_rows_sentiment.csv")
df['date'] = pd.to_datetime(df['unix_time_seconds'], unit='s')

# Initialize query engine
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

# Sentiment Analysis Section
st.title('Sentiment Analysis Over Time')
binning_option = st.selectbox('Select time binning:', options=['6h', '1h', '2h', '12h', '1D'], index=0)
time_binned_sentiment = df.resample(binning_option, on='date').sentiment.mean().dropna()
st.line_chart(time_binned_sentiment)

# Chat Section
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.write("Chat with Gamebeast, you can ask for insights about what players are saying about your game")
    user_message = st.chat_input("Ask Gamebeast about Players:")

    if user_message:
        st.session_state.conversation_history.append(("user", user_message))
        system_response = query_engine.query(user_message).response
        st.session_state.conversation_history.append(("assistant", system_response))

    chat_container = st.container(height=500)
    for role, message in st.session_state.conversation_history:
        sender = "User" if role == "user" else "Gamebeast"
        chat_container.chat_message(sender).write(message)
