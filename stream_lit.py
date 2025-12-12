# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load .env file
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# LangChain imports (same ones you used)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import StrOutputParser

# ---------------------------------------------------------------------
# STREAMLIT APP CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="CRP-CXL Conservation Chatbot",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 CRP-CXL Conservation Chatbot")
st.write("Ask any question about conservation solutions, interventions, and more")

# ---------------------------------------------------------------------
# LLM + Embeddings (unchanged)
# ---------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# ---------------------------------------------------------------------
# Load FAISS (unchanged)
# ---------------------------------------------------------------------
DB_PATH = "faiss_index"
db = FAISS.load_local("DB_PATH", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# ---------------------------------------------------------------------
# PROMPTS (unchanged except for var name = input)
# ---------------------------------------------------------------------
reformulation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite the user's latest message into a standalone question. Do not answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a conservation expert for answering conservation related solutions \
            Use the following pieces of retrieved context to answer the question \
            If you don't know the answer, just say that you don't know \
            Use ten sentences maximum and keep the answer concise. \

            {context}"""),
        ("human", "{input}")
    ]
)

# ---------------------------------------------------------------------
# RAG PIPELINE (UNMODIFIED)
# ---------------------------------------------------------------------
retriever_with_history = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=reformulation_prompt
)

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
    document_variable_name="context"
)

rag_chain = create_retrieval_chain(
    retriever_with_history,
    question_answer_chain
)

from langchain.schema.runnable import RunnableLambda

def extract_answer(rag_output):
    # rag_output looks like: {"context": [...], "input": "...", "answer": "the real text"}
    return rag_output["answer"]

pipeline = rag_chain | RunnableLambda(extract_answer) | StrOutputParser()


# ---------------------------------------------------------------------
# STREAMLIT CHAT STATE
# ---------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------------------
user_input = st.chat_input("Ask something about conservation...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        output_placeholder = st.empty()
        full_response = ""

    try:
        # FIX: only pass {"input": user_input}
        for chunk in pipeline.stream({"input": user_input}):
            full_response += chunk
            output_placeholder.markdown(full_response)

    except Exception as e:
        full_response = f"⚠️ Error: {e}"
        output_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


        # Append to chat_history (LangChain wants list of dicts or messages)
        # st.session_state.chat_history.append({"role": "user", "content": user_input})
        # st.session_state.chat_history.append({"role": "assistant", "content": full_response})