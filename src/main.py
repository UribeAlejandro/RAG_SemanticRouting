from dotenv import load_dotenv

from src import logger
import streamlit as st
from src.serving.utils import get_app, stream_response

load_dotenv()


if __name__ == "__main__":
    app = get_app()

    st.header("LLM Documentation", divider="rainbow")
    st.caption("""Welcome to the RAG with semantic routing. 
    Ask me anything about LLM (Agents, Adversarial Attacks, Prompt Engineering). 
    I can also answer questions not related to LLMs""")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Your question here"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        inputs = {"question":prompt}
        with st.chat_message("assistant"):
            for output in app.stream(inputs):
                for key, value in output.items():
                    logger.info("Finished running: %s:", key)
            response = value["generation"]
            st.write_stream(stream_response(response))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
