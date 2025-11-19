# app.py
import os
from dotenv import load_dotenv, find_dotenv

# Load .env
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# LangChain / OpenAI imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser

# Chainlit
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig

# ---------- Setup LLM, Embeddings, Vector DB ----------
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# Load existing FAISS index
DB_PATH = "../faiss_index"
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# ---------- Prompts ----------
# Reformulation prompt
reformulation_system = """Given the chat history and a recent user question,
generate a standalone question that can be answered without the chat history.
DO NOT answer it — only reformulate it."""
reformulation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", reformulation_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# QA prompt
qa_system_prompt = """
You are an assistant that answers user questions using the retrieved context below.

{context}

Use the context to answer the question. Keep answers concise (max 6 sentences).
If the answer is not present, say "I don't know".
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)

# ---------- RAG Chains ----------
retriever_with_history = create_history_aware_retriever(
    llm, retriever, reformulation_prompt
)

# IMPORTANT FIX: unify variable names
question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
    document_variable_name="context",
    input_variable_name="question"
)

rag_chain = create_retrieval_chain(
    retriever_with_history,
    question_answer_chain
)

# ---------- Chainlit Handlers ----------
@cl.on_chat_start
async def on_chat_start():
    runnable = rag_chain | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    await cl.Message(content="Hello! Ask me anything.").send()


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    if not runnable:
        await cl.Message(content="Chat pipeline not initialized. Restart the chat.").send()
        return

    response_msg = cl.Message(content="")

    try:
        # IMPORTANT FIX: correct async streaming call
        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(
                callbacks=[cl.LangchainCallbackHandler()]
            )
        ):
            await response_msg.stream_token(chunk)

    except Exception as e:
        await response_msg.send()
        await cl.Message(content=f"Error: {e}").send()
        return

    await response_msg.send()
