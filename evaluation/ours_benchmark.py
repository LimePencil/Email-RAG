import os
import json
import pandas as pd
import time
import argparse
import re
from collections import Counter
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
from tqdm import tqdm

# Constants
ELASTIC_CLOUD_ID = ##
ELASTIC_API_KEY = ##

tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0", data_dir="/data/taeho/self-rag/model")

def setup_environment():
    os.environ.update({
        "OPENAI_API_KEY": #
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

    document_content_description = "KAIST academic email data."
    print("Creating vector store...")
    vectorstore = ElasticsearchStore(
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
        index_name="elasticsearch-self-query_for_demo3",
        es_cloud_id=ELASTIC_CLOUD_ID,
        es_api_key=ELASTIC_API_KEY
    )
    print("finish vector store...")
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o", frequency_penalty=1.0)
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
        ("박종철 교수님이 황태호에게 답장한 이메일의 내용은 무엇인가요?", {"query": "박종철 교수님이 황태호에게 답장한 이메일의 내용", "filter": 'and(like("from", "%박종철%"), like("to", "%황태호%"))'},),
        ("2024년 6월 30일 이전에 열리는 Data Intelligence Workshop 2024에 사전 등록하려면 어떻게 해야 하나요?",{"query": "Data Intelligence Workshop 2024 조기등록","filter": '"and(lte("year", "2024-12-31"), lte("month", "6"), lte("day", "30"))"'})
    ]

def setup_chat_chain(retriever, emails):
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o", frequency_penalty=1.0)
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
        """오늘은 2024년 8월 8일 오후 7시입니다. 제 이름은 황태호입니다. 제 이메일은 doubleyyh@kaist.ac.kr입니다. 질문을 제 개인 정보를 고려하여 자연스럽게 같은 의미의 질문으로 다시 작성해 주세요. 설명이나 구조화 없이 답만 간결하게 하세요.\n\nInput: {question}
        """
    )
    
    search_query = create_search_query(llm, condense_question_prompt, date_time_question_prompt)
    
    def chat(message, history):
        start_time = time.time()
        cnt = 0
        while cnt < 5:
            try:
                new_query = search_query.invoke({"chat_history": history, "question": message})
                query_time = time.time() - start_time
                
                results_docs = retriever.invoke(new_query)
                retrieval_time = time.time() - start_time - query_time
                
                context = generate_context(results_docs, emails)
                context_generation_time = time.time() - start_time - query_time - retrieval_time
                
                history_langchain_format = format_chat_history(history)
                
                for _ in range(5):
                    response = chain.invoke({
                        "message": message, 
                        "context": context,
                        "history": history_langchain_format
                    })
                    gc_result = UpstageGroundednessCheck().invoke({"context": context, "answer": response})
                    if gc_result.lower().startswith("grounded"):
                        break
                
                total_time = time.time() - start_time
                
                return {
                    "response": response,
                    "metrics": {
                        "query_time": query_time,
                        "retrieval_time": retrieval_time,
                        "context_generation_time": context_generation_time,
                        "total_time": total_time
                    }
                }
            except Exception as e:
                print(e)
                cnt += 1
                if cnt == 5:
                    return {
                        "response": "I don't know.",
                        "metrics": {
                            "query_time": 0,
                            "retrieval_time": 0,
                            "context_generation_time": 0,
                            "total_time": 0
                        }
                    }
            
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
    original_uid = [doc.metadata["uid"] for doc in results_docs]
    uid_set = list(set(original_uid))
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

def qa_f1_score(prediction, ground_truth):
    prediction_tokens = tokenize_korean(normalize_answer(prediction))
    ground_truth_tokens = tokenize_korean(normalize_answer(ground_truth))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s):
    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punctuation(lower(s)))

def tokenize_korean(text):
    return tokenizer.tokenize(text)

def main(email_file: str, input_file: str, output_file: str):
    setup_environment()
    emails = load_and_preprocess_emails(email_file)
    mails = []
    retriever = setup_retriever(mails)
    chat = setup_chat_chain(retriever, emails)
    
    queries = json.load(open(input_file))
    
    results = []
    for query_data in tqdm(queries[:100]):
        result = chat(query_data["questions"], [])
        f1 = qa_f1_score(result["response"], query_data["answers"])
        results.append({
            "query": query_data["questions"],
            "response": result["response"],
            "ground_truth": query_data["answers"],
            "f1_score": f1,
            "metrics": result["metrics"]
        })
        # break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmarking results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email RAG System Benchmarking")
    parser.add_argument("--email_file", type=str, help="Path to the input JSON file containing email data", default="/data/taeho/Email-RAG/cleaned_email_data.json")
    parser.add_argument("--input_file", type=str, default="/data/taeho/email_questions_longform.json")
    parser.add_argument("--output_file", type=str, help="Path to save the benchmarking results", default="/data/taeho/email_rag_benchmark_results_longform.json")
    args = parser.parse_args()
    
    main(args.email_file, args.input_file ,args.output_file)