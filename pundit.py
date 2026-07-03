from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END

from config import get_chat_model
from retrieve import get_retriever
from chain import format_docs
from store import load_store

MAX_STEPS = 10


class PlanDecision(BaseModel):
    decision: Literal["CONTINUE", "DONE"] = Field(
        description="CONTINUE if more retrieval is needed to write a "
                    "well-grounded take; DONE if enough has been gathered."
    )
    next_query: str = Field(
        description="If CONTINUE: the search query to run next. "
                    "If DONE: an empty string."
    )


class PunditState(TypedDict):
    topic: str
    gathered: List[str]
    next_query: str
    decision: str
    steps: int
    verdict: str
    stalled: bool


def plan_node(state: PunditState) -> dict:
    model = get_chat_model()
    planner = model.with_structured_output(PlanDecision)

    gathered_text = "\n\n".join(state["gathered"]) or "(nothing gathered yet)"

    prompt = (
        f"You are planning research for an analytical take on: {state['topic']}\n\n"
        f"Context gathered so far:\n{gathered_text}\n\n"
        "Decide: is this enough to write a well-grounded analytical take, "
        "or do you need one more targeted retrieval? "
        "If more is needed, provide the next search query. "
        "If nothing has been gathered yet, you should CONTINUE with an "
        "opening query about the topic. If the newly gathered context is not adding new information beyond what you already have, choose DONE"
    )

    decision_obj = planner.invoke(prompt)

    return {
        "decision": decision_obj.decision,
        "next_query": decision_obj.next_query,
        "steps": state["steps"] + 1,
    }


def act_node(state: PunditState) -> dict:
    retriever = get_retriever(
        search_type="similarity_score_threshold",
        k=4, score_threshold=0.5, store=load_store(),
    )

    docs = retriever.invoke(state["next_query"])
    new_context = format_docs(docs)
    if new_context in state["gathered"]:
        return {"stalled": True}
    return {"gathered": state["gathered"] + [new_context], "stalled": False}


def write_node(state: PunditState) -> dict:
    model = get_chat_model()
    gathered_text = "\n\n".join(state["gathered"]) or "(no records gathered)"

    pundit_prompt = (
        f"You are writing an analytical take on: {state['topic']}\n\n"
        f"Context gathered:\n{gathered_text}\n\n"
        "Find the single most surprising, uncomfortable, or overlooked pattern in the "
        "records -- the one angle a bold, committed take could be built "
        "around. It has to be a pattern actually present in the records, not invented "
        "and not a judgment: 'wins ugly, sixteen fouls a game' is a pattern; 'doesn't "
        "deserve to win' is a judgment, and judgments are off-limits.\n\n"
        "Now, write the finished take in a few punchy sentences, fully committed to that one angle. "
        "Do not summarize. Do not list multiple facts. Build the whole take around the one angle."
    )

    statement = model.invoke(pundit_prompt)
    return {"verdict": statement.content}


def route(state: PunditState) -> str:
    if state["steps"] >= MAX_STEPS or state["decision"] == "DONE" or state.get("stalled"):
        return "write"
    else:
        return "act"


def build_pundit():
    builder = StateGraph(PunditState)
    builder.add_node("plan", plan_node)
    builder.add_node("act", act_node)
    builder.add_node("write", write_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route, ["act", "write"])
    builder.add_edge("act", "plan")
    builder.add_edge("write", END)

    return builder.compile()


if __name__ == "__main__":
    agent = build_pundit()

    initial: PunditState = {
        "topic": "France vs Norway",
        "gathered": [],
        "next_query": "",
        "decision": "",
        "steps": 0,
        "verdict": "",
    }

    for update in agent.stream(initial, stream_mode="updates"):
        print(update)
        print("---")

    print("\n=== FINAL VERDICT ===")
    print(agent.invoke(initial)["verdict"])
