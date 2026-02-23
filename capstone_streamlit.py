from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, List
import streamlit as st

load_dotenv()

@st.cache_resource
def load_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found!")

    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

@st.cache_resource
def load_embedding():
    api_key = os.getenv("OPENROUTER_API_KEY")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

llm = load_llm()
embedding = load_embedding()

vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 20})

class RecState(Dict):
    user_query: str
    preferences: str
    retrieved_docs: List
    recommendations: str

def extract_preferences(state: RecState):

    prompt = f"""
    Analyze the user's book preferences based on their message:

    "{state['user_query']}"

    Extract the following:
    - what the user liked
    - what the user disliked
    - desired themes, tropes, tones, vibes, genres
    - specific elements they want more or less of

    Output a concise preference profile.
    """

    response = llm.invoke(prompt)
    state["preferences"] = response
    return state

def retrieve_books(state: RecState):

    query = f"""
    Find books that match these user preferences:

    {state['preferences']}
    """

    docs = retriever.invoke(query)
    
    seen = set()
    unique_docs = []
    for doc in docs: 
        book_id = doc.metadata["book_id"]
        if book_id not in seen:
            unique_docs.append(doc)
            seen.add(book_id)

    state["retrieved_docs"] = unique_docs
    return state

def generate_recommendations(state: RecState):

    user_query = state["user_query"]
    context = "\n\n".join([d.page_content for d in state["retrieved_docs"]])

    prompt = f"""
    USER QUERY:
    {user_query}

    PREFERENCE ANALYSIS:
    {state['preferences']}

    RELEVANT BOOK DATA:
    {context}

    Based on this information, recommend 3-5 books.
    For each recommendation:
    - explain why it matches the user’s preferences ("Why it matches:")
    - highlight specific themes, elements, or vibes ("Themes and elements:")
    Keep the tone friendly and helpful.
    """

    response = llm.invoke(prompt)
    state["recommendations"] = response
    return state

rag_graph = (
    StateGraph(RecState)
    .add_node("extract", extract_preferences)
    .add_node("retrieve", retrieve_books)
    .add_node("recommend", generate_recommendations)
    .set_entry_point("extract")
    .add_edge("extract", "retrieve")
    .add_edge("retrieve", "recommend")
    .add_edge("recommend", END)
    .compile()
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #c5dae2;
        padding: 1rem;
    }
    """,
    unsafe_allow_html=True
)

st.title("🐉Fantasy Book Recommender📚")
query = st.text_input("Explain what you like or dislike about Fantasy books. You could also mention your favorite books.")

if query:
    with st.spinner('Generating answer...be patient! :)'):
        result = rag_graph.invoke({"user_query": query})
    st.subheader("Answer")
    st.write(result["recommendations"].content)
