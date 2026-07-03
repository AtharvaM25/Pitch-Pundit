import os
import dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

dotenv.load_dotenv()


def get_chat_model():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file.")
        return ChatGroq(model="llama-3.3-70b-versatile")

    elif provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file.")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'gemini'.")


def get_embedding_model():
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file.")
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. Use 'gemini' or 'ollama'.")


if __name__ == "__main__":
    chat_model = get_chat_model()
    reply = chat_model.invoke("Hello World?")
    print(reply.content)

    embeddings = get_embedding_model()
    vector = embeddings.embed_query("Mexico beat South Africa")
    print(f"embedding ok: {len(vector)} dimensions")
