import os
os.environ["GOOGLE_API_KEY"]=st.secrets["GOOGLE_API_KEY"]


import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

# App title
st.title("🤖 Chat with Gemini 2.0 Flash")

# User input
user_input = st.text_input("You:", key="user_input")

# Submit button
if st.button("Send") and user_input.strip():
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get model response
    result = llm.invoke(st.session_state.chat_history)

    # Append AI response
    st.session_state.chat_history.append(AIMessage(content=result.content))

    # Clear input
    st.session_state.user_input = ""

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.markdown(f"**You:** {message.content}")
    elif isinstance(message, AIMessage):
        st.markdown(f"**AI:** {message.content}")


