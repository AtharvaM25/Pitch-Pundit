from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import get_chat_model
from retrieve import get_retriever
from chain import format_docs, build_prompt
from store import load_store


def build_reformulation_prompt() -> ChatPromptTemplate:
    system_msg = (
        "Given the conversation history and the user's latest question, "
        "rewrite the latest question as a STANDALONE question that can be "
        "understood without needing the history.\n\n"
        "Rules:\n"
        "1. If the question refers back to something earlier (e.g. 'what about "
        "the semifinal?', 'and their next game?', 'how many did he score?'), "
        "resolve those references using the history so the question stands "
        "on its own.\n"
        "2. If the question is ALREADY standalone, return it EXACTLY as-is, "
        "unchanged.\n"
        "3. Output ONLY the rewritten question. Do NOT answer it. Do NOT add "
        "explanation, preamble, or quotes. Your entire response is a single "
        "question."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_reformulator():
    new_prompt = build_reformulation_prompt()
    model = get_chat_model()
    reformulate = new_prompt | model | StrOutputParser()

    def choose(x):

        if not x.get("chat_history"):
            return x["input"]
        else:
            return reformulate.invoke(x)

    return RunnableLambda(choose)


def build_history_aware_chain(retriever=None):
    if retriever is None:
        retriever = get_retriever(
            search_type="similarity_score_threshold",
            k=4, score_threshold=0.5, store=load_store(),
        )

    reformulator = build_reformulator()
    prompt = build_prompt()
    model = get_chat_model()

    chain = (RunnablePassthrough.assign(standalone=reformulator) | {
        "context": (lambda x: x["standalone"]) | retriever | format_docs,
        "input": lambda x: x["standalone"],
    } | prompt | model | StrOutputParser())

    return chain


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage

    reformulator = build_reformulator()

    fake_history = [
        HumanMessage(content="How did Mexico's opening match go?"),
        AIMessage(content="Mexico won 2-1 in their opener [Source 1]."),
    ]

    print("=== reformulation in isolation ===")
    print("no history, standalone Q (should pass through UNCHANGED):")
    print("  ->", reformulator.invoke(
        {"input": "Who won the opening match?", "chat_history": []}
    ))
    print("\nwith history, follow-up (should become STANDALONE):")
    print("  ->", reformulator.invoke(
        {"input": "What about the semifinal?", "chat_history": fake_history}
    ))

    chain = build_history_aware_chain()
    print("\n=== full chain ===")
    print("A:", chain.invoke(
        {"input": "What about their semifinal?", "chat_history": fake_history}
    ))
