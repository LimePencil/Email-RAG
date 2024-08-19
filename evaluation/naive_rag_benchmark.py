import os
import json
import pandas as pd
import time
import argparse
import re
import string
from collections import Counter
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import UpstageEmbeddings, ChatUpstage, UpstageGroundednessCheck
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema import AIMessage, HumanMessage
from transformers import AutoTokenizer
from tqdm import tqdm
# Constants
ELASTIC_CLOUD_ID = #
ELASTIC_API_KEY = #

tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0", data_dir="/data/taeho/self-rag/model")


def tokenize_korean(text):
    return tokenizer.tokenize(text)


def setup_environment():
    os.environ.update({
        "OPENAI_API_KEY": #
        "NEO4J_URI": #
        "NEO4J_USERNAME": #
        "NEO4J_PASSWORD": #
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
    vectorstore = ElasticsearchStore(
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
        index_name="elasticsearch-self-query_for_demo3",
        es_cloud_id=ELASTIC_CLOUD_ID,
        es_api_key=ELASTIC_API_KEY
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k":20})
    return retriever

def setup_chat_chain(retriever, emails):
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for KAIST academic email question-answering tasks. Use the following pieces of retrieved email content to answer the question considering the history of the conversation. If you don't know the answer, just say that you don't know. Do not any explanation.\n---\nCONTEXT:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ])
    
    chain = rag_prompt | ChatUpstage() | StrOutputParser()
    
    def chat(message, history):
        start_time = time.time()
        
        results_docs = retriever.invoke(message)
        retrieval_time = time.time() - start_time
        
        context = generate_context(results_docs, emails)
        context_generation_time = time.time() - start_time - retrieval_time
        
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
                "retrieval_time": retrieval_time,
                "context_generation_time": context_generation_time,
                "total_time": total_time
            }
        }

    return chat

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

def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

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