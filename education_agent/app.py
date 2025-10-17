import streamlit as st
import asyncio
from agent import call_agent  # Your async function to call the agent

st.set_page_config(page_title="🎓 Education Chatbot", layout="centered")
st.title("🎓 Education Insights Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about the education dataset...")

if user_query:
    # Display user's message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the async agent function
            response = asyncio.run(call_agent(user_query))
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
