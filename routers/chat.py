#version of using graph rag

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

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core import pydantic_v1
from langchain_core.pydantic_v1 import Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_upstage import UpstageEmbeddings, ChatUpstage, UpstageGroundednessCheck
from langchain.chains import LLMChain

from dotenv import load_dotenv
from typing import Optional

load_dotenv()


# Model definitions
class Entities(pydantic_v1.BaseModel):
    names: List[str] = Field(
        ...,
        description="All the person, organization, location, time, numbers, email or entities that "
        "appear in the text e.g) John Doe, KAIST, Apple Inc., 사회적 감정에 대한 실험, Floid IoT 실험, NLP*CL 연구실, 피시험자 번호, 2023년 6월 23일, 2023-06-23, 2023/06/23, 2023.06.23, doubleyy@kaist.ac.kr, ME251"
    )
class Question(BaseModel):
    history: List[Tuple[str, str]]
    question: str
class Answer(BaseModel):
    answer:str

# Initialize components
def init_components():
    vector_index = Neo4jVector.from_existing_graph(
        UpstageEmbeddings(model="solar-embedding-1-large"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    solar = ChatUpstage()
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    graph = Neo4jGraph()
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    return vector_index, solar, llm, graph

# Entity extraction
def create_entity_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Use the given format to extract information from the following input: {question}"),
    ])
    return prompt | llm.with_structured_output(Entities)

# Structured retriever
def structured_retriever(question: str, entity_chain, graph):
    entities = entity_chain.invoke({"question": question})
    print("entities", entities)

    new_response = []
    meta_data = {}
    body_text = {}

    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r]->(neighbor)
              WHERE NOT type(r) = 'MENTIONS' AND NOT neighbor:Source
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r]-(neighbor)
              WHERE NOT type(r) = 'MENTIONS' AND NOT neighbor:Source
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )

        response2 = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:1})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[:!MENTIONS]-(source:Source)
              WITH source
              MATCH (source)-[r]-()
              WHERE NOT type(r) = 'MENTIONS'
              RETURN DISTINCT 'source:\n' + id(source) + '\n' + startNode(r).id + ' - ' + type(r) + ' -> ' + endNode(r).id AS output
                UNION ALL
                WITH node
              MATCH (node)<-[:!MENTIONS]-(source:Source)
              WITH source
              MATCH (source)-[r]-()
              WHERE NOT type(r) = 'MENTIONS'
              RETURN DISTINCT 'source:\n' + id(source) + '\n' + startNode(r).id + ' - ' + type(r) + ' -> ' + endNode(r).id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )

        response = list(set([r["output"] for r in response]))
        response2 = list(set([r["output"] for r in response2]))
        response += response2

        for r in response:
            if 'source' in r:
                doc_id = int(r.split("source:\n")[1].split("\n")[0])
                if doc_id not in meta_data:
                    meta_data[doc_id] = {}
                if not body_text.get(doc_id):
                    body_text[doc_id] = r.split("source:\n")[1].split("\n")[1].split(" -> ")[0].strip()
                
                relations = {
                    'SENT_TO': ('to', 'to'),
                    'CC_TO': ('cc', 'cc'),
                    'HAS_SUBJECT': ('subject', 'subject'),
                    'SENT_BY': ('from', 'from'),
                    'SENT_ON_DATE': ('date', 'date'),
                    'SENT_AT_TIME': ('time', 'time')
                }

                for relation, (key, value) in relations.items():
                    if relation in r:
                        if key not in meta_data[doc_id]:
                            meta_data[doc_id][key] = []
                        meta_data[doc_id][key].append(r.split(f"{relation} ->")[1].strip())
            else:
                new_response.append(r)
        
    for key, value in body_text.items():
        t_body = ""
        for field in ['from', 'to', 'cc']:
            if field in meta_data[key]:
                t_body += f"{field.capitalize()}: {', '.join(meta_data[key][field])}\n"
        for field in ['subject', 'date', 'time']:
            if field in meta_data[key]:
                t_body += f"{field.capitalize()}: {meta_data[key][field][0]}\n"
        t_body += f"Content: {value}\n"
        body_text[key] = t_body

    result = "\n".join(new_response)
    return result, body_text

# Summarize chain
def create_summarize_chain(llm):
    template = "Given these relations:\n{relations}\n\nPlease make a concise natural language form summary. Do not any explanation:"
    prompt = PromptTemplate(template=template, input_variables=["relations"])
    return LLMChain(llm=llm, prompt=prompt)

# Retriever
def retriever(question: str, vector_index, entity_chain, graph, summarize_chain):
    print(f"Search query: {question}")
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question, k=10)]
    structured_data, body_text = structured_retriever(question, entity_chain, graph)
    print(structured_data)
    
    if structured_data:
        summary = summarize_chain.run(relations=structured_data)
        final_data = f"{summary}\n"
    else:
        final_data = ""
    
    for value in body_text.values():
        final_data += f"\n{value}"
    final_data += "-" * 10
    final_data += "\n".join(unstructured_data)
    print(final_data)
    return final_data

# Condense question
def create_condense_question_prompt():
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    return PromptTemplate.from_template(template)

def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    return [
        message
        for human, ai in chat_history
        for message in [HumanMessage(content=human), AIMessage(content=ai)]
    ]

# Search query
def create_search_query(llm, condense_question_prompt):
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: format_chat_history(x["chat_history"])
            )
            | condense_question_prompt
            | llm
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x : x["question"]),
    )

# QA Chain
def create_qa_chain(solar, search_query, retriever_func):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise. Use Korean.
    Answer:"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an kaist-domain email qa chatbot. The content of qa is mainly focused on the academic things. User name is 황태호, and today's date is 5, August, 2024."),
            MessagesPlaceholder(variable_name="history"),
            ("human",template)
        ]
    )

    retriever_chain = RunnableParallel(
        {
            "context": search_query | retriever_func,
            "question": RunnablePassthrough(),
        }
    )

    chain = prompt | llm | StrOutputParser()

    return retriever_chain, chain

def qa_chain(question: str,history, retriever_chain, chain, groundedness_check):
    retrieval = retriever_chain.invoke({"question": question})
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    for _ in range(5):
        answer = chain.invoke(retrieval|{"history":history_langchain_format})
        gc_result = groundedness_check.invoke({"context": retrieval["context"], "answer": answer})
        if gc_result.lower().startswith("grounded"):
            print("✅ Groundedness check passed")
            break
        else:
            print("❌ Groundedness check failed")

    return answer



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


vector_index, solar, llm, graph = init_components()
entity_chain = create_entity_chain(llm)
summarize_chain = create_summarize_chain(llm)
condense_question_prompt = create_condense_question_prompt()
search_query = create_search_query(llm, condense_question_prompt)
retriever_func = lambda question: retriever(question, vector_index, entity_chain, graph, summarize_chain)
retriever_chain, chain = create_qa_chain(llm, search_query, retriever_func)
groundedness_check = UpstageGroundednessCheck()

@router.get("/",response_class=HTMLResponse)
async def get_test(request: Request):
    return templates.TemplateResponse('index.html',{"request":request})

@router.post("/ask")
async def ask_question(question:Question):
    answer = qa_chain(question.question,question.history,retriever_chain,chain,groundedness_check)
    return {"answer" : answer}
 
