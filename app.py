import streamlit as st
import requests

st.set_page_config(page_title="Visa Assistant", page_icon="🌍")

st.title("🌍 AI Visa Assistant (Powered by RAG API)")

API_URL = "http://127.0.0.1:5000/chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

# display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
if prompt := st.chat_input("Ask about visa requirements..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"query": prompt}
                )

                data = response.json()
                answer = data.get("answer", "No response")

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })