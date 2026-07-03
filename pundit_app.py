import streamlit as st
from pundit import build_pundit

st.set_page_config(page_title="Pitch Pundit — Agent", page_icon="🎙️")
st.title("🎙️ Pitch Pundit — Autonomous Pundit Agent")
st.caption(
    "Give it a topic. The agent plans its own research, gathers match records "
    "over several steps, and writes a grounded analytical take — watch it work."
)


@st.cache_resource
def get_agent():
    return build_pundit()


agent = get_agent()

topic = st.text_input(
    "Topic",
    placeholder="e.g. France vs Norway, ask about a match",
)
go = st.button("Generate take", type="primary")

if go and topic:
    initial = {
        "topic": topic,
        "gathered": [],
        "next_query": "",
        "decision": "",
        "steps": 0,
        "verdict": "",
        "stalled": False,
    }

    final_verdict = None

    with st.status("Researching…", expanded=True) as status:
        for update in agent.stream(initial, stream_mode="updates"):
            node = list(update.keys())[0]
            payload = update[node]
            if node == "plan":
                st.write(
                    f"🧠 Planning — {payload['decision']}: {payload['next_query']}")
            elif node == "act":
                st.write("🔍 Retrieved match records")
            elif node == "write":
                final_verdict = payload["verdict"]

        status.update(label="Take ready", state="complete")

    if final_verdict:
        st.subheader("The Take")
        st.markdown(final_verdict)
    else:
        st.write("No verdict captured.")

elif go and not topic:
    st.warning("Enter a topic first.")
