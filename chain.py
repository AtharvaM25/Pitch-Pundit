from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from store import load_store

from config import get_chat_model
from retrieve import get_retriever


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant match data was found in the tournament records."

    blocks = []
    for i, doc in enumerate(docs, 1):
        blocks.append(f"[Source {i}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_prompt() -> ChatPromptTemplate:
    system_msg = (
        "You are a football analyst for the FIFA World Cup 2026. You answer "
        "questions using only the official match records provided to you.\n\n"
        "Rules you must follow:\n"
        "1. Answer using ONLY the match data given below. Do not use any "
        "outside knowledge, memory of past tournaments, or assumptions.\n"
        "2. If the match data does not contain the answer, say so plainly — "
        "for example: \"That isn't in the match records I have.\" Never guess, "
        "never invent scores, scorers, dates, or details.\n"
        "3. When you state a fact, refer to the source it came from using its "
        "[Source N] label.\n"
        "4. Be concise and factual. You are reporting from records, not "
        "speculating."
    )

    human_msg = (
        "Match data:\n"
        "{context}\n\n"
        "Question: {input}"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", human_msg),
        ]
    )


def build_chain(retriever=None):
    if retriever is None:
        retriever = get_retriever(search_type="similarity_score_threshold",
                                  k=4, score_threshold=0.5, store=load_store())

    prompt = build_prompt()
    model = get_chat_model()

    setup_dict = {}
    setup_dict["context"] = retriever | format_docs
    setup_dict["input"] = RunnablePassthrough()

    chain = setup_dict | prompt | model | StrOutputParser()
    return chain


def ask(question: str, chain=None) -> str:
    if chain is None:
        chain = build_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    chain = build_chain()

    questions = [
        "How did Mexico's opening match go?",
        "Who won the 2022 World Cup final?",
        "What was the score in the final?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q, chain=chain)}")
