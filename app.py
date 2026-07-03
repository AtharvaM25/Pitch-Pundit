import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from history_chain import build_history_aware_chain

st.set_page_config(
    page_title="Pitch Pundit — Ask the Tournament", page_icon="⚽")
st.title("⚽ Pitch Pundit — Ask the Tournament")
st.caption(
    "Grounded Q&A on the FIFA World Cup 2026. Answers come only from official match records.")


@st.cache_resource
def get_chain():
    return build_history_aware_chain()


chain = get_chain()


if "history" not in st.session_state:
    st.session_state.history = []


for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


user_input = st.chat_input("Ask about a match, team, or result…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking the records…"):
            answer = chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.history,
            })
        st.markdown(answer)

    st.session_state.history.append(HumanMessage(content=user_input))
    st.session_state.history.append(AIMessage(content=answer))
