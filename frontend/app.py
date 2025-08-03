import streamlit as st
import requests
import random 
import time

def get_rag_response(prompt):
  url = "http://127.0.0.1:5000/get"
  try:
    r = requests.post(url, json={"message": prompt})
    r.raise_for_status()
    return r.json().get("response", "")
  
  except Exception as e:
    return f"Error: {str(e)}"

def stream_response(prompt):
  response = get_rag_response(prompt)  
  for word in response.split():
    yield word + " "
    time.sleep(0.03)


st.markdown("""
    <div style="text-align: center; padding-top: 10px;">
        <img src="https://img.icons8.com/ios-filled/100/5C6BC0/stethoscope.png" width="50"/>
        <h1 style="display: inline; margin-left: 10px; color: #3E4EB8; font-family: 'Segoe UI', sans-serif;">
            CareBot
        </h1>
        <p style="color: #6c757d; font-size: 16px; margin-top: 5px;">
            Your Personal Medical Assistant
        </p>
    </div>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
  st.session_state.messages = []
  st.session_state.messages.append({'role': 'assistant', 'content': """
Hi! I'm CareBot, your AI-powered medical assistant, here to help you understand symptoms, medications, and general health questions.

Feel free to ask questions like:
- _"I have a sore throat and fever — what should I do?"_
- _"What are the side effects of ibuprofen?"_

Let’s take care of your health, one question at a time.
"""})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])
  
if prompt := st.chat_input("Type here"):
  st.chat_message('user').markdown(prompt)
  st.session_state.messages.append({'role': 'user', 'content': prompt})

  with st.chat_message('assistant'):
    with st.spinner("Thinking..."):
      # write_stream() requires a callable function 
      full_response = st.write_stream(stream_response(prompt))
  st.session_state.messages.append({'role': 'assistant', 'content': full_response}) 

