import os 
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file  

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=OPENAI_API_KEY)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

db = FAISS.load_local("../faiss_index", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"),allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":6})
from langchain import hub

prompt=hub.pull("rlm/rag-prompt")
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """ Given the chat history and a recent user question \
    generate a standalone question \
    that can be answered without the context of the chat history. DO NOT answer the question, \
    just reformulate it if needed or otherwise return it as is. """
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

retriever_with_history = create_history_aware_retriever(
llm, retriever, prompt
)

# pip install chainlit
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import chainlit as cl

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=OPENAI_API_KEY)
    system_prompt = """ Given the chat history and a recent user question \
    generate a standalone question \
    that can be answered without the context of the chat history. DO NOT answer the question, \
    just reformulate it if needed or otherwise return it as is. """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    qa_system_prompt = """
    You are an assistant for question answering.
    Use the following retrieved context to answer the user's question.

    {context}

    Keep answers concise (max 6 sentences). 
    If the answer is not available, say 'I don't know'.
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "{input}")
    ])
    
    retriever_with_history = create_history_aware_retriever(
    llm, retriever, prompt
    )
    
    # chat_history = []
    # qa_system_prompt = """You are an assistant for question answering tasks \
    # Use the following pieces of retrieved context to answer the question \
    # If you don't know the answer, just say that you don't know \
    # Use six senetences maximum and keep the answer concise. \
    # {context}"""
    
    # qa_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", qa_system_prompt),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}")
    #     ]
    # )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


    rag_chain=create_retrieval_chain(retriever_with_history, question_answer_chain)
    
    runnable = prompt | model | rag_chain | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
        
            # Example of how you might pass the context

    await msg.send()

# @cl.on_chat_start
# async def on_chat_start():
#     qa_system_prompt = """You are an assistant for question answering tasks \
#     Use the following pieces of retrieved context to answer the question \
#     If you don't know the answer, just say that you don't know \
#     Use six senetences maximum and keep the answer concise. \

#     {context}"""
#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", qa_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}")
#         ]
#     )

#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#     rag_chain=create_retrieval_chain(retriever_with_history, question_answer_chain)
#     cl.user_session.set("rag_chain", rag_chain)

# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain=create_retrieval_chain(retriever_with_history, question_answer_chain)

# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cl.user_session.get("rag_chain")  # type: Runnable

#     msg = cl.Message(content="")

#     for chunk in await cl.make_async(runnable.stream)(
#         {"question": message.content},
#         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#     ):
#         await msg.stream_token(chunk)

#     await msg.send()
# import textwrap 
# from langchain_core.messages import HumanMessage
# chat_history = []
# question = "What are effective solutions for reducing agricultural runoff?"
# ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
# chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])
# print(textwrap.fill(ai_msg_1["answer"], width=100))
# second_question = "Does it help reduce extinction of aquatic ecosystems?"

# ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
# print(textwrap.fill(ai_msg_2["answer"], width=100))
# third_question = "What countries will benefit from these solutions?"

# ai_msg_3 = rag_chain.invoke({"input": third_question, "chat_history": chat_history})
# print(textwrap.fill(ai_msg_3["answer"], width=100))
# fourth_question = "Which endangered species would be beneficial?"

# ai_msg_4 = rag_chain.invoke({"input": fourth_question, "chat_history": chat_history})
# print(textwrap.fill(ai_msg_4["answer"], width=100))