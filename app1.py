# app1.py
import os
from dotenv import load_dotenv, find_dotenv

# Load .env
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Langchain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import StrOutputParser

# Chainlit
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig

# LLM + embeddings
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# Load FAISS
DB_PATH = "../faiss_index"
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# -------------------------------------------------------------------------
# PROMPTS — ONLY `input` IS ALLOWED.
# -------------------------------------------------------------------------

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
        "Use the context below to answer the question.\n\n{context}\n\n"
        "If unsure, say 'I don't know'. Keep answers under 6 sentences."),
        ("human", "{input}")
    ]
)

# -------------------------------------------------------------------------
# RAG CHAINS
# -------------------------------------------------------------------------

retriever_with_history = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=reformulation_prompt
)

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
    document_variable_name="context",
)

rag_chain = create_retrieval_chain(
    retriever_with_history,
    question_answer_chain
)

# -------------------------------------------------------------------------
# CHAINLIT HANDLERS
# -------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    runnable = rag_chain | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    await cl.Message(content="Hello! Ask me anything.").send()

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    response_msg = cl.Message(content="")

    try:
        async for chunk in runnable.astream(
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
        ):
            await response_msg.stream_token(chunk)

    except Exception as e:
        await response_msg.send()
        await cl.Message(content=f"Error: {e}").send()
        return

    await response_msg.send()
