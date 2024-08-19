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

import pickle
from transformers import AutoTokenizer
from ner import recognize_address, recognize_name
from korean_name import KoreanName
tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0")


def load_and_preprocess_emails(file_path: str) -> pd.DataFrame:
    emails = json.load(open(file_path))
    df = pd.DataFrame(emails)
    return df[df['text_body'] != '']

def get_or_add_id(entity, entity_list) -> int:

    if entity in entity_list:
        return entity_list.index(entity)
    else:
        entity_list.append(entity)
        return len(entity_list)

def split_emails(emails: pd.DataFrame) -> List[Document]:
    mails = []
    from tqdm import tqdm
    tqdm.pandas()
    names = [] # store full name in default, and if there is no full name, store first name temporarily
    addresses = []

    for _, email in tqdm(emails.iterrows(), total=emails.shape[0]):
        t_subject = email['subject']
        t_date, t_time = parse_date_time(email['date'])
        t_cc, t_from, t_to = parse_recipients(email)
        page_content = f"제목: {t_subject}\n내용: {email['text_body']}"

        for i,(name,address) in enumerate(t_cc):
            name_i, address_i = None, None
            if(name and KoreanName.is_korean_fullname(name)):
                name_i = get_or_add_id(name,names)
            if(address):
                address_i = get_or_add_id(address,addresses)
            t_cc[i] = [f"이름@{name_i}" if name_i else name,f"주소@{address_i}" if address_i else address]
        for i,(name,address) in enumerate(t_from):
            name_i, address_i = None, None
            if(name and KoreanName.is_korean_fullname(name)):
                name_i = get_or_add_id(name,names)
            if(address):
                address_i = get_or_add_id(address,addresses)
            t_from[i] = [f"이름@{name_i}" if name_i else name,f"주소@{address_i}" if address_i else address]
        for i,(name,address) in enumerate(t_to):
            name_i, address_i = None, None
            if(name and KoreanName.is_korean_fullname(name)):
                name_i = get_or_add_id(name,names)
            if(address):
                address_i = get_or_add_id(address,addresses)
            t_to[i] = [f"이름@{name_i}" if name_i else name,f"주소@{address_i}" if address_i else address]

        person_list = recognize_name(page_content).person_list
        for person in person_list:
            exists = False
            for i,name in enumerate(names):
                if person.name == name or person.name == name[1:]:
                    exists = True
                    address_i = i+1
                    break
            if(not exists):
                names.append(person.name)
                address_i = len(names)

            page_content = page_content.replace(person.name, f"이름@{address_i}")
            if(person.name in t_subject):
                t_subject = t_subject.replace(person.name, f"이름@{address_i}")

        address_list = recognize_address(page_content).address_list
        for address in address_list:
            address_i = get_or_add_id(address.address,addresses)
            page_content = page_content.replace(address.address,f"주소@{address_i}")

            if(address.address in t_subject):
                t_subject = t_subject.replace(address.address, f"주소@{address_i}")

        for comb in tqdm(generate_combinations(t_from, t_to, t_cc)):
            mail = create_document(page_content, t_subject, comb, t_date, t_time, email['uid'])
            mails.append(mail)
            break
    return mails, names, addresses

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
    # truncate the page_content to 4000 tokens
    encoded_page_content = tokenizer.encode(page_content, max_length=3500, truncation=True)
    page_content = tokenizer.decode(encoded_page_content)
    return Document(page_content=page_content, metadata=metadata)


def upload_embedding_and_entity_lookuptable(mails,names,addresses):
    uri = os.getenv('uri')

    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client.email
    entity_collection = db['entity']
    address_collection = db['address']

    name_documents = [{"name":name,"id":i+1} for i,name in enumerate(names)]
    if name_documents:
        entity_collection.insert_many(name_documents)
    
    address_documents = [{"address":address,"id":i+1} for i,address in enumerate(addresses)]
    if address_documents:
        address_collection.insert_many(address_documents)

    print("Creating vector store...")
    vectorstore = ElasticsearchStore.from_documents(
        mails[-100000:],
        UpstageEmbeddings(model="solar-embedding-1-large"),
        index_name="elasticsearch-self-query_for_demo4_deidentified",
        es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
        es_api_key=os.getenv("ELASTIC_API_KEY"), 
    )
    print("finish vector store...")


# def setup_retriever(mails):
#     metadata_field_info = [
#         AttributeInfo(name="subject", description="The subject of the email", type="string"),
#         AttributeInfo(name="cc", description="The CC recipients of the email. e.g.) doubleyyh@kaist.ac.kr, 대학원 총학생회, 총장, 학생#전체, 학술문화관 DB담당자, 황태호, 신명금", type="string"),
#         AttributeInfo(name="from", description="The sender of the email. e.g.) doubleyyh@kaist.ac.kr, 대학원 총학생회, 총장, 학생#전체, 학술문화관 DB담당자, 신명금, 황태호", type="string"),
#         AttributeInfo(name="to", description="The recipient of the email. e.g.) doubleyyh@kaist.ac.kr, 대학원 총학생회, 총장, 학생#전체, 학술문화관 DB담당자, 신명금", type="string"),
#         AttributeInfo(name="year", description="The year the email was sent. e.g.) 2022", type="integer"),
#         AttributeInfo(name="month", description="The month the email was sent. e.g.) 01, 02", type="integer"),
#         AttributeInfo(name="day", description="The day the email was sent. e.g.) 01, 05, 28, 31", type="integer"),
#         AttributeInfo(name="hour", description="The hour the email was sent. e.g.) 12, 23, 22", type="integer"),
#         AttributeInfo(name="minute", description="The minute the email was sent. e.g.) 00, 05, 12", type="integer"),
#         AttributeInfo(name="second", description="The second the email was sent. e.g.) 00, 34, 25, 59", type="integer"),
#     ]

#     document_content_description = "KAIST academic email data"
    
#     vectorstore = ElasticsearchStore(
#         embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
#         index_name="elasticsearch-self-query_for_demo3",
#         es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
#         es_api_key=os.getenv("ELASTIC_API_KEY")
#     )
    
#     llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
#     query_constructor = get_query_constructor_prompt(
#         document_content_description,
#         metadata_field_info,
#         examples=get_query_examples()
#     ) | llm | StructuredQueryOutputParser.from_components()

#     from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator
#     return SelfQueryRetriever(
#         query_constructor=query_constructor,
#         vectorstore=vectorstore,
#         structured_query_translator=ElasticsearchTranslator(),
#         search_kwargs={"k": 20},
#     )

# def get_query_examples():
#     return [
#         ("DB담당자가 보낸 요가매트 관련된 메일을 찾아줘.", {"query": "요가매트", "filter": 'like("from", "%DB담당자%")'}),
#         ("학생#전체가 보낸 세미나 메일 알려줘.", {"query": "세미나 메일 알려줘.", "filter": 'contain("from", "학생")'}),
#         ("제목에 대학원 총학생회가 들어간 메일 알려줘.", {"query": "대학원 총학생회", "filter": 'like("subject", "대학원%총%학생회")'}),
#         ('오늘은 2024년 8월 8일이야. 오늘 전에 온 도서관에 관련된 메일 알려줘.', {"query": "도서관에 관련된 메일 알려줘.", "filter": 'and(lt("year", 2024), or(lt("month", 8), and(eq("month", 8), lt("day", 8))))'}),
#         ('나는 황태호야. 내가 2023년 10월 26일에 보낸 KCC 2024 학회 관련 메일 알려줘.', {"query": "KCC 2024 학회 관련 메일 알려줘.", "filter": 'and(eq("from", "황태호"), eq("year", 2023), eq("month", 10), eq("day", 26))'}),
#     ]

# def setup_chat_chain(retriever, emails):
#     llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
#     rag_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are an assistant for KAIST academic email question-answering tasks. Use the following pieces of retrieved email content to answer the question considering the history of the conversation. If you don't know the answer, just say that you don't know. \n---\nCONTEXT:\n{context}"),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{message}"),
#     ])
    
#     chain = rag_prompt | ChatUpstage() | StrOutputParser()
    
#     condense_question_prompt = PromptTemplate.from_template(
#         "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question (decontextualize), in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
#     )
    
#     date_time_question_prompt = PromptTemplate.from_template(
#         """오늘은 2024년 8월 8일 오후 7시입니다. 제 이름은 황태호입니다. 제 이메일은 doubleyyh@kaist.ac.kr입니다. 질문을 제 개인 정보를 고려하여 자연스럽게 같은 의미의 질문으로 다시 작성해 주세요. 설명이나 구조화 없이 답만 간결하게 하세요.\n\nexample:
#     Input: "내가 받은 메일을 알려줘."
#     Output: "황태호가 받은 메일을 알려줘."
# example:
#     Input: "최근 5일간 받은 메일을 알려줘."
#     Output: "2024년 08월 03일부터 2024년 08월 08일까지 황태호가 받은 메일을 알려줘."
# example:
#     Input: "오늘 오후 3시에 받은 메일을 알려줘."
#     Output: "2024년 08월 08일 오후 03시에 황태호가 받은 메일을 알려줘."
# example:
#     Input: "어제 받은 메일을 알려줘."
#     Output: "2024년 08월 07일 황태호가 받은 메일을 알려줘."
# example:
#     Input: "최근 받은 메일을 알려줘."
#     Output: "2024년 08월 05일부터 2024년 08월 08일까지 황태호가 받은 메일을 알려줘."
# example:
#     Input: "대학원 총학생회에서 온 메일을 알려줘."
#     Output: "2024년 8월 8일 기준으로 가장 가까운 대학원 총학생회에서 온 메일을 알려줘."
# example:
#     Input: "제목에 대학원 총학생회가 들어간 메일 알려줘."
#     Output: "제목에 대학원 총학생회가 들어간 메일 알려줘."
# example:
#     Input: "DB담당자가 보낸 요가매트 관련된 메일을 찾아줘."
#     Output: "DB담당자가 보낸 요가매트 관련된 메일을 찾아줘."
# example:
#     Input: "학생#전체가 보낸 세미나 메일 알려줘."
#     Output: "학생#전체가 보낸 세미나 메일 알려줘."
# example:
#     Input: "제목 [도서관] 특허 검색 A to Z 알려드립니다라는 메일은. 언제왔어?"
#     Output: "제목 '[도서관]특허 검색 A to Z 알려드립니다'라는 메일은 언제왔어?"
# \n\nInput: {question}
# """
#     )
    
#     search_query = create_search_query(llm, condense_question_prompt, date_time_question_prompt)
    
#     def chat(message, history):
#         new_query = search_query.invoke({"chat_history": history, "question": message})
#         results_docs = retriever.invoke(new_query)
#         # print(results_docs)
#         context = generate_context(results_docs, emails)
#         print(context)
#         # tokenize the context max_length 2500



#         history_langchain_format = format_chat_history(history)
        
#         for _ in range(5):
#             response = chain.invoke({
#                 "message": message, 
#                 "context": context,
#                 "history": history_langchain_format
#             })
#             gc_result = UpstageGroundednessCheck().invoke({"context": results_docs, "answer": response})
#             if gc_result.lower().startswith("grounded"):
#                 print("✅ Groundedness check passed")
#                 return response
#             print("❌ Groundedness check failed")
#         return response

#     return chat

# def create_search_query(llm, condense_question_prompt, date_time_question_prompt):
#     return RunnableBranch(
#         (
#             RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
#             RunnablePassthrough.assign(chat_history=lambda x: format_chat_history(x["chat_history"]))
#             | condense_question_prompt
#             | llm
#             | StrOutputParser()
#             | date_time_question_prompt
#             | llm
#             | StrOutputParser(),
#         ),
#         RunnableLambda(lambda x : x["question"])
#         | date_time_question_prompt
#         | llm
#         | StrOutputParser(),
#     )

# def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
#     buffer = []
#     for human, ai in chat_history:
#         buffer.append(HumanMessage(content=human))
#         buffer.append(AIMessage(content=ai))
#     return buffer

# def generate_context(results_docs, emails):
#     # ordered set of unique uids
#     original_uid = [doc.metadata["uid"] for doc in results_docs]
#     uid_set = list(set(original_uid))
#     # sort by original order
#     uid_set.sort(key=original_uid.index)
#     context = ""
#     email_form = "From: {froms}\nTo: {tos}\nCC: {ccs}\nDate: {date}\nSubject: {subject}\n\n{body}\n------\n\n"
#     for uid in uid_set:
#         email = emails.loc[emails['uid'] == uid].iloc[0]
#         context += email_form.format(
#             froms=email['from'], tos=email['to'], ccs=email['cc'],
#             date=email['date'], subject=email['subject'], body=email['text_body']
#         )
#     encoded_context = tokenizer.encode(context, max_length=2000, truncation=True)
#     truncated_context = tokenizer.decode(encoded_context, skip_special_tokens=True)
#     return truncated_context

def main():

    load_dotenv()
    current_dir = os.path.dirname(__file__)
    file_name = "cleaned_email_data_v4.json"
    # data 폴더의 JSON 파일 경로를 생성
    file_path = os.path.join(current_dir, '..', 'data', 'graph_rag', file_name)
    emails = load_and_preprocess_emails(file_path)
    mails,names,addresses = split_emails(emails)
    with open('mails_list.pkl','wb') as f:
        pickle.dump(mails,f)
    with open('names_list.pkl','wb') as f:
        pickle.dump(names,f)
    with open('addresses_list.pkl','wb') as f:
        pickle.dump(addresses,f)
    upload_embedding_and_entity_lookuptable(mails,names,addresses)
    # retriever = setup_retriever(mails)
    # chat = setup_chat_chain(retriever, emails)
    
    # Example usage
    # response = chat("제목 '특허 검색'이 들어간 메일은 언제왔어?", [])
    # print(response)

if __name__ == "__main__":
    main()