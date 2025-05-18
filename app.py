import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import os
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Set up Streamlit page
st.set_page_config(page_title="Gemini Chatbot", layout="centered")
st.title("🤖 Gemini Chatbot with Memory")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

# Display past messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)

# User input
user_input = st.chat_input("Say something...")
if user_input:
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)

    # Get response from Gemini
    result = llm.invoke(st.session_state.chat_history)
    response = result.content

    # Append AI response
    st.session_state.chat_history.append(AIMessage(content=response))
    st.chat_message("assistant").markdown(response)
    
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
result = llm.invoke("how is breast cancer classified?")
print("AI:",result.content)  
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

loader = TextLoader('https://arxiv.org/pdf/2410.01755')
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(data)
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
vector_store = FAISS.from_documents(docs,embeddings)
ret = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
result = ret.invoke("how is breast cancer classified?")['result']
result
