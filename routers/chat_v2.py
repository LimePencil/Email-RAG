#version of using self-quering

import sys
sys.path.append("..")

from starlette import status
from starlette.responses import RedirectResponse

from fastapi import Depends, APIRouter, Request, Form
# import models
# from database import engine, SessionLocal
# from sqlalchemy.orm import Session
# from .auth import get_current_user

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Body
from pydantic import BaseModel
from typing import List, Tuple


import os
import json
import pandas as pd
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings, ChatUpstage, UpstageGroundednessCheck
from langchain_elasticsearch import ElasticsearchStore
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQueryOutputParser
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import AIMessage, HumanMessage
from langchain.chains.query_constructor.base import (
        StructuredQueryOutputParser,
        get_query_constructor_prompt,
    )

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv
from typing import Optional

load_dotenv()
# Constants

uri = os.getenv('uri')

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# use a database named "myDatabase"
db = client.email

# use a collection named "recipes"
my_collection = db["email"]

class Question(BaseModel):
    history: List[Tuple[str, str]]
    question: str
class Answer(BaseModel):
    answer:str

def load_and_preprocess_emails(file_path: str) -> pd.DataFrame:
    emails = json.load(open(file_path))
    df = pd.DataFrame(emails)
    return df[df['text_body'] != '']

def split_emails(emails: pd.DataFrame) -> List[Document]:
    mails = []
    for _, email in emails.iterrows():
        t_subject = email['subject']
        t_date, t_time = parse_date_time(email['date'])
        t_cc, t_from, t_to = parse_recipients(email)
        page_content = f"제목: {t_subject}\n내용: {email['text_body']}"
        
        for comb in generate_combinations(t_from, t_to, t_cc):
            mail = create_document(page_content, t_subject, comb, t_date, t_time, email['uid'])
            mails.append(mail)
    return mails

def parse_date_time(date_str: str) -> Tuple[str, str]:
    if not date_str:
        return "", ""
    date, time = date_str.split("T")
    return date[1:], time[:-1]

def parse_recipients(email: pd.Series) -> Tuple[List, List, List]:
    return email.get('cc', []), email.get('from', []), email.get('to', [])

def generate_combinations(t_from, t_to, t_cc):
    import itertools
    for_iter_from = [tt for t in t_from for tt in t if tt] or [""]
    for_iter_to = [tt for t in t_to for tt in t if tt] or [""]
    for_iter_cc = [tt for t in t_cc for tt in t if tt] or [""]
    return list(itertools.product(for_iter_from, for_iter_to, for_iter_cc))

def create_document(page_content, subject, comb, date, time, uid):
    metadata = {
        "subject": subject, "cc": comb[2], "from": comb[0], "to": comb[1],
        "year": 9999, "month": 99, "day": 99, "hour": 99, "minute": 99, "second": 99, "uid": uid
    }
    if date and time:
        metadata.update({
            "year": int(date.split("-")[0]), "month": int(date.split("-")[1]), "day": int(date.split("-")[2]),
            "hour": int(time.split(":")[0]), "minute": int(time.split(":")[1]), "second": int(time.split(":")[2])
        })
    return Document(page_content=page_content, metadata=metadata)

def setup_retriever():
    metadata_field_info = [
        AttributeInfo(name="subject", description="The subject of the email", type="string"),
        AttributeInfo(name="cc", description="The CC recipients of the email. e.g.) doubleyyh@kaist.ac.kr, 대학원 총학생회, 총장, 학생#전체, 학술문화관 DB담당자, 황태호, 신명금", type="string"),
        AttributeInfo(name="from", description="The sender of the email. e.g.) doubleyyh@kaist.ac.kr, 대학원 총학생회, 총장, 학생#전체, 학술문화관 DB담당자, 신명금, 황태호", type="string"),
        AttributeInfo(name="to", description="The recipient of the email. e.g.) doubleyyh@kaist.ac.kr, 대학원 총학생회, 총장, 학생#전체, 학술문화관 DB담당자, 신명금", type="string"),
        AttributeInfo(name="year", description="The year the email was sent. e.g.) 2022", type="integer"),
        AttributeInfo(name="month", description="The month the email was sent. e.g.) 01, 02", type="integer"),
        AttributeInfo(name="day", description="The day the email was sent. e.g.) 01, 05, 28, 31", type="integer"),
        AttributeInfo(name="hour", description="The hour the email was sent. e.g.) 12, 23, 22", type="integer"),
        AttributeInfo(name="minute", description="The minute the email was sent. e.g.) 00, 05, 12", type="integer"),
        AttributeInfo(name="second", description="The second the email was sent. e.g.) 00, 34, 25, 59", type="integer"),
    ]

    document_content_description = "KAIST academic email data"
    
    vectorstore = ElasticsearchStore(
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
        index_name="elasticsearch-self-query_for_demo3",
        es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
        es_api_key=os.getenv("ELASTIC_API_KEY")
    )

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o",frequency_penalty=1.0)
    query_constructor = get_query_constructor_prompt(
        document_content_description,
        metadata_field_info,
        examples=get_query_examples()
    ) | llm | StructuredQueryOutputParser.from_components()

    from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator
    return SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        structured_query_translator=ElasticsearchTranslator(),
        search_kwargs={"k": 50},
    )

def get_query_examples():
    return [
        ("류석영 교수님이 보낸 세미나 관련된 메일을 찾아줘.", {"query": "세미나 관련된 메일", "filter": 'like("from", "%류석영%")'}),
        ("DB담당자님이 보낸 요가매트 관련된 메일을 찾아줘.", {"query": "요가매트 관련된 메일", "filter": 'like("from", "%DB담당자%")'}),
        ("류석영 교수님이 2023년 5월 1일부터 2023년 7월 31일까지 보낸 세미나 관련 메일을 찾아줘.", {
"query": "세미나 관련 메일을 찾아줘.",
"filter": 'and(like("from", "%류석영%"), gte("year", 2023), lte("year", 2023), gte(and("month", 5), "day", 1), lte(and("month", 7), "day", 31))'
}),
        ("DB담당자가 보낸 요가매트 관련된 메일을 찾아줘.", {"query": "요가매트 관련된 메일", "filter": 'like("from", "%DB담당자%")'}),
        ("DB담당자한테서 온 요가매트 관련된 메일을 찾아줘.", {"query": "요가매트 관련된 메일", "filter": 'like("from", "%DB담당자%")'}),
        ("학생#전체가 보낸 세미나 메일 알려줘.", {"query": "세미나 메일", "filter": 'like("from", "%학생%")'}),
        ("김현수 교수님이 보낸 제목에 '연구'가 포함된 메일을 모두 찾아줘.", {"query": "연구 관련된 메일을 찾아줘.", "filter": 'and(like("from", "%김현수%"), like("subject", "%연구%"))'}),
        ("2023년 3월부터 2023년 6월까지 학생#전체가 보낸 제목에 '강연'이나 '워크샵'이 포함된 메일을 찾아줘.", {
    "query": "강연 워크샵",
    "filter": 'and(like("from", "%학생#전체%"), gte("year", 2023), lte("year", 2023), gte(and("month", 3), "day", 1), lte(and("month", 6), "day", 30), or(like("subject", "%강연%"), like("subject", "%워크샵%")))'}),
        ("제목에 '대학원 총학생회'가 들어간 메일 알려줘.", {"query": "대학원 총학생회 관련된 메일을 찾아줘.", "filter": 'like("subject", "%대학원%총%학생회%")'}),
        ("제목에 혁신 교육이 들어간 메일 알려줘.", {"query": "혁신 교육 관련된 메일을 알려줘.", "filter": 'like("subject", "%혁신%교육%")'}),
        ('오늘은 2024년 8월 8일이야. 오늘 전에 온 도서관에 관련된 메일 알려줘.', {"query": "도서관에 관련된 메일 알려줘.", "filter": 'and(lt("year", 2024), or(lt("month", 8), and(eq("month", 8), lt("day", 8))))'}),
        ('나는 황태호야. 내가 2023년 10월 26일에 보낸 KCC 2024 학회 관련 메일 알려줘.', {"query": "KCC 2024 학회 관련 메일 알려줘.", "filter": 'and(like("from", "%황태호%"), eq("year", 2023), eq("month", 10), eq("day", 26))'}),
        ("박종철 교수님이 황태호에게 답장한 이메일의 내용은 무엇인가요?", {"query": "박종철 교수님이 황태호에게 답장한 이메일의 내용", "filter": 'and(like("from", "%박종철%"), like("to", "%황태호%"))'},)
    ]

def setup_chat_chain(retriever):
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o", frequency_penalty=1.0)
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for KAIST academic email question-answering tasks. My name is 황태호. My email address is doubleyyh@kaist.ac.kr. Today is 08.07. 7PM. Use the following pieces of retrieved email content to answer the question considering the history of the conversation. If you don't know the answer, just say that you don't know. \n---\nCONTEXT:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ])
    
    chain = rag_prompt | llm | StrOutputParser()
    
    condense_question_prompt = PromptTemplate.from_template(
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question (decontextualize), in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
    )
    
    date_time_question_prompt = PromptTemplate.from_template(
        """오늘은 2024년 8월 8일 오후 7시입니다. 제 이름은 황태호입니다. 제 이메일은 doubleyyh@kaist.ac.kr입니다. 질문을 제 개인 정보를 고려하여 자연스럽게 같은 의미의 질문으로 다시 작성해 주세요. 날짜를 고려할 필요가 없다면 고려하지 말아주세요. 설명이나 구조화 없이 답만 간결하게 하세요.\n\nexample:
    Input: "내가 받은 메일을 알려줘."
    Output: "황태호가 받은 메일을 알려줘."
example:
    Input: "최근 5일간 받은 메일을 알려줘."
    Output: "2024년 08월 03일부터 2024년 08월 08일까지 황태호가 받은 메일을 알려줘."
example:
    Input: "오늘 오후 3시에 받은 메일을 알려줘."
    Output: "2024년 08월 08일 오후 03시에 황태호가 받은 메일을 알려줘."
example:
    Input: "어제 받은 메일을 알려줘."
    Output: "2024년 08월 07일 황태호가 받은 메일을 알려줘."
example:
    Input: "최근 받은 메일을 알려줘."
    Output: "2024년 08월 05일부터 2024년 08월 08일까지 황태호가 받은 메일을 알려줘."
example:
    Input: "대학원 총학생회에서 온 메일을 알려줘."
    Output: "2024년 8월 8일 기준으로 가장 가까운 대학원 총학생회에서 온 메일을 알려줘."
example:
    Input: "제목에 대학원 총학생회가 들어간 메일 알려줘."
    Output: "제목에 대학원 총학생회가 들어간 메일 알려줘."
example:
    Input: "DB담당자가 보낸 요가매트 관련된 메일을 찾아줘."
    Output: "DB담당자가 보낸 요가매트 관련된 메일을 찾아줘."
example:
    Input: "학생#전체가 보낸 세미나 메일 알려줘."
    Output: "학생#전체가 보낸 세미나 메일 알려줘."
example:
    Input: "제목 [도서관] 특허 검색 A to Z 알려드립니다라는 메일은. 언제왔어?"
    Output: "제목 '[도서관]특허 검색 A to Z 알려드립니다'라는 메일은 언제왔어?"
example:
    Input: "업스테이지 단기 강의 관련 메일 알려줘."
    Output: "업스테이지 단기 강의 관련 메일 알려줘."
\n\nInput: {question}
"""
    )
    
    search_query = create_search_query(llm, condense_question_prompt, date_time_question_prompt)
    
    def chat(message, history):
        new_query = search_query.invoke({"chat_history": history, "question": message})
        results_docs = retriever.invoke(new_query)
        print(results_docs)
        context = generate_context(results_docs)
        print(context)
        history_langchain_format = format_chat_history(history)
        
        for _ in range(5):
            response = chain.invoke({
                "message": message, 
                "context": context,
                "history": history_langchain_format
            })
            gc_result = UpstageGroundednessCheck().invoke({"context": context, "answer": response})
            if gc_result.lower().startswith("grounded"):
                print("✅ Groundedness check passed")
                return response
            print("❌ Groundedness check failed")
        return response

    return chat

def create_search_query(llm, condense_question_prompt, date_time_question_prompt):
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
            RunnablePassthrough.assign(chat_history=lambda x: format_chat_history(x["chat_history"]))
            | condense_question_prompt
            | llm
            | StrOutputParser()
            # | date_time_question_prompt
            # | llm
            # | StrOutputParser()
        ),
        RunnableLambda(lambda x : x["question"])
        # | date_time_question_prompt
        # | llm
        # | StrOutputParser(),
    )

def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer
    
def generate_context(results_docs):
    # ordered set of unique uids
    original_uid = [doc.metadata["uid"] for doc in results_docs]
    uid_set = list(set(original_uid))
    # sort by original order
    uid_set.sort(key=original_uid.index)
    context = ""
    email_form = "From: {froms}\nTo: {tos}\nCC: {ccs}\nDate: {date}\nSubject: {subject}\n\n{body}\n------\n\n"
    for uid in uid_set:
        email = my_collection.find_one({"uid": uid})
        context += email_form.format(
            froms=email['from'], tos=email['to'], ccs=email['cc'],
            date=email['date'], subject=email['subject'], body=email['text_body']
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0", data_dir="/data/taeho/self-rag/model")
    encoded_context = tokenizer.encode(context, max_length=4000, truncation=True)
    truncated_context = tokenizer.decode(encoded_context)
    
    return truncated_context


retriever = setup_retriever()
chat = setup_chat_chain(retriever)


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}}
)

#models.Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="templates")


# def get_db():
#     try:
#         db = SessionLocal()
#         yield db
#     finally:
#         db.close()

@router.get("/",response_class=HTMLResponse)
async def get_test(request: Request):
    return templates.TemplateResponse('index.html',{"request":request})

@router.post("/ask")
async def ask_question(question:Question):
    answer = chat(question.question,question.history)
    return {"answer" : answer}