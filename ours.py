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
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0", data_dir="/data/taeho/self-rag/model")
# Constants
ELASTIC_CLOUD_ID = #
ELASTIC_API_KEY = #

def setup_environment():
    os.environ.update({
        "OPENAI_API_KEY":#,
        "UPSTAGE_API_KEY": #
    })

def load_and_preprocess_emails(file_path: str) -> pd.DataFrame:
    emails = json.load(open(file_path))
    df = pd.DataFrame(emails)
    return df[df['text_body'] != '']

def split_emails(emails: pd.DataFrame) -> List[Document]:
    mails = []
    from tqdm import tqdm
    tqdm.pandas()

    for _, email in tqdm(emails.iterrows(), total=emails.shape[0]):
        t_subject = email['subject']
        t_date, t_time = parse_date_time(email['date'])
        t_cc, t_from, t_to = parse_recipients(email)
        page_content = f"제목: {t_subject}\n내용: {email['text_body']}"
        
        for comb in tqdm(generate_combinations(t_from, t_to, t_cc)):
            mail = create_document(page_content, t_subject, comb, t_date, t_time, email['uid'])
            mails.append(mail)
            break
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
    # truncate the page_content to 4000 tokens
    encoded_page_content = tokenizer.encode(page_content, max_length=3500, truncation=True)
    page_content = tokenizer.decode(encoded_page_content)
    return Document(page_content=page_content, metadata=metadata)

def setup_retriever(mails):
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
    print("Creating vector store...")
    # vectorstore = ElasticsearchStore.from_documents(
    #     mails[-100000:],
    #     UpstageEmbeddings(model="solar-embedding-1-large"),
    #     index_name="elasticsearch-self-query_for_demo3",
    #     es_cloud_id=ELASTIC_CLOUD_ID,
    #     es_api_key=ELASTIC_API_KEY, 
    # )
    vectorstore = ElasticsearchStore(
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
        index_name="elasticsearch-self-query_for_demo3",
        es_cloud_id=ELASTIC_CLOUD_ID,
        es_api_key=ELASTIC_API_KEY
    )
    print("finish vector store...")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
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
        ("류석영 교수님이 보낸 세미나 관련된 메일을 찾아줘.", {"query": "세미나", "filter": 'like("from", "%류석영%")'}),
        ("DB담당자님이 보낸 요가매트 관련된 메일을 찾아줘.", {"query": "요가매트", "filter": 'like("from", "%DB담당자%")'}),
        ("DB담당자가 보낸 요가매트 관련된 메일을 찾아줘.", {"query": "요가매트", "filter": 'like("from", "%DB담당자%")'}),
        ("DB담당자한테서 온 요가매트 관련된 메일을 찾아줘.", {"query": "요가매트", "filter": 'like("from", "%DB담당자%")'}),
        ("DB담당자가 보낸 요가매트 관련된 메일을 찾아줘.", {"query": "요가매트", "filter": 'like("from", "%DB담당자%")'}),
        ("학생#전체가 보낸 세미나 메일 알려줘.", {"query": "세미나 메일 알려줘.", "filter": 'like("from", "%학생%")'}),
        ("제목에 '대학원 총학생회'가 들어간 메일 알려줘.", {"query": "대학원 총학생회", "filter": 'like("subject", "대학원%총%학생회")'}),
        ("제목에 혁신 교육이 들어간 메일 알려줘.", {"query": "혁신 교육", "filter": 'like("subject", "대학원%총%학생회")'}),
        ('오늘은 2024년 8월 8일이야. 오늘 전에 온 도서관에 관련된 메일 알려줘.', {"query": "도서관에 관련된 메일 알려줘.", "filter": 'and(lt("year", 2024), or(lt("month", 8), and(eq("month", 8), lt("day", 8))))'}),
        ('나는 황태호야. 내가 2023년 10월 26일에 보낸 KCC 2024 학회 관련 메일 알려줘.', {"query": "KCC 2024 학회 관련 메일 알려줘.", "filter": 'and(like("from", "%황태호%"), eq("year", 2023), eq("month", 10), eq("day", 26))'}),
    ]

def setup_chat_chain(retriever, emails):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for KAIST academic email question-answering tasks. Use the following pieces of retrieved email content to answer the question considering the history of the conversation. If you don't know the answer, just say that you don't know. Do not any explanation.\n---\nCONTEXT:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ])
    
    chain = rag_prompt | ChatUpstage() | StrOutputParser()
    
    condense_question_prompt = PromptTemplate.from_template(
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question (decontextualize), in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
    )
    
    date_time_question_prompt = PromptTemplate.from_template(
        """오늘은 2024년 8월 8일 오후 7시입니다. 제 이름은 황태호입니다. 제 이메일은 doubleyyh@kaist.ac.kr입니다. 질문을 제 개인 정보를 고려하여 자연스럽게 같은 의미의 질문으로 다시 작성해 주세요. 설명이나 구조화 없이 답만 간결하게 하세요.\n\nexample:
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
\n\nInput: {question}
"""
    )
    
    search_query = create_search_query(llm, condense_question_prompt, date_time_question_prompt)
    
    def chat(message, history):
        new_query = search_query.invoke({"chat_history": history, "question": message})
        results_docs = retriever.invoke(new_query)
        # print(results_docs)
        context = generate_context(results_docs, emails)
        print(context)
        # tokenize the context max_length 2500



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
            | date_time_question_prompt
            | llm
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x : x["question"])
        | date_time_question_prompt
        | llm
        | StrOutputParser(),
    )

def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

def generate_context(results_docs, emails):
    # ordered set of unique uids
    original_uid = [doc.metadata["uid"] for doc in results_docs]
    uid_set = list(set(original_uid))
    # sort by original order
    uid_set.sort(key=original_uid.index)
    context = ""
    email_form = "From: {froms}\nTo: {tos}\nCC: {ccs}\nDate: {date}\nSubject: {subject}\n\n{body}\n------\n\n"
    for uid in uid_set:
        email = emails.loc[emails['uid'] == uid].iloc[0]
        context += email_form.format(
            froms=email['from'], tos=email['to'], ccs=email['cc'],
            date=email['date'], subject=email['subject'], body=email['text_body']
        )
    encoded_context = tokenizer.encode(context, max_length=4000, truncation=True)
    truncated_context = tokenizer.decode(encoded_context, skip_special_tokens=True)
    return truncated_context

def main():
    setup_environment()
    emails = load_and_preprocess_emails("/data/taeho/Email-RAG/cleaned_email_data.json")
    # mails = split_emails(emails)
    mails = []
    retriever = setup_retriever(mails)
    chat = setup_chat_chain(retriever, emails)
    
    # Example usage
    response = chat("업스테이지 관련된 메일을 보여줘.", [])
    print(response)
    print()

if __name__ == "__main__":
    main()