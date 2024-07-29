import os
import json
import pandas as pd
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_upstage import UpstageEmbeddings
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_core.documents.base import Document

def setup_environment():
    os.environ["OPENAI_API_KEY"] = "sk-"
    os.environ["NEO4J_URI"] = ""
    os.environ["NEO4J_USERNAME"] = ""
    os.environ["NEO4J_PASSWORD"] = ""
    os.environ["UPSTAGE_API_KEY"] = "up-"

def load_and_preprocess_emails(file_path):
    emails = json.load(open(file_path))
    df = pd.DataFrame(emails)
    return df[df['text_body'] != '']

def split_emails(emails, text_splitter):
    mails = []
    cc, received_from, froms, tos, dates, times, texts, subjects = [], [], [], [], [], [], [], []
    
    for _, email in emails.iterrows():
        t_subject = email['subject']
        t_date = email['date'].split("T")[0][1:]
        t_time = email['date'].split("T")[1][:-1]
        t_cc = email['cc']
        t_from = email['from']
        t_to = email['to']
        
        splitted = text_splitter.create_documents([email['text_body']])
        for split in splitted:
            subjects.append(t_subject)
            cc.append(t_cc)
            froms.append(t_from)
            tos.append(t_to)
            dates.append(t_date)
            times.append(t_time)
            mail = Document(page_content=f"{split.page_content}")
            mail = text_splitter.split_documents([mail])
            texts.append(mail[0])
    
    return texts, subjects, cc, froms, tos, dates, times

def add_node(graph, id, type):
    graph.nodes.append(Node(id=id, type=type))

def add_relationship(graph, source_id, source_type, target_id, target_type, rel_type):
    graph.relationships.append(Relationship(
        source=Node(id=source_id, type=source_type),
        target=Node(id=target_id, type=target_type),
        type=rel_type
    ))

def create_graph_documents(texts, subjects, cc, froms, tos, dates, times, llm_transformer):
    graph_documents = llm_transformer.convert_to_graph_documents(texts)
    
    for i, g in enumerate(graph_documents):
        content = texts[i].page_content
        date = dates[i]
        time = times[i]
        subject = subjects[i]

        add_node(g, date, "Date")
        add_node(g, time, "Time")
        add_node(g, subject, "Subject")
        add_node(g, content, "Source")

        add_relationship(g, content, "Source", date, "Date", "SENT_ON_DATE")
        add_relationship(g, content, "Source", time, "Time", "SENT_AT_TIME")
        add_relationship(g, content, "Source", subject, "Subject", "HAS_SUBJECT")

        for sender in froms[i]:
            add_node(g, sender[1], "Email")
            add_relationship(g, content, "Source", sender[1], "Email", "SENT_BY")
            
            if sender[0]:
                add_node(g, sender[0], "Person")
                add_relationship(g, sender[0], "Person", sender[1], "Email", "HAS_EMAIL")
                add_relationship(g, sender[1], "Email", sender[0], "Person", "HAS_NAME")
            
            add_relationship(g, sender[1], "Email", date, "Date", "SENT_ON_DATE")
            add_relationship(g, sender[1], "Email", time, "Time", "SENT_AT_TIME")

            for recipient in tos[i] + cc[i]:
                add_node(g, recipient[1], "Email")
                rel_type = "SENT_TO" if recipient in tos[i] else "CC_TO"
                add_relationship(g, content, "Source", recipient[1], "Email", rel_type)

                if recipient[0]:
                    add_node(g, recipient[0], "Person")
                    add_relationship(g, recipient[0], "Person", recipient[1], "Email", "HAS_EMAIL")
                    add_relationship(g, recipient[1], "Email", recipient[0], "Person", "HAS_NAME")

                add_relationship(g, sender[1], "Email", recipient[1], "Email", rel_type)
                add_relationship(g, recipient[1], "Email", sender[1], "Email", "RECEIVED_FROM")
                add_relationship(g, recipient[1], "Email", date, "Date", "RECEIVED_ON_DATE")
                add_relationship(g, recipient[1], "Email", time, "Time", "RECEIVED_AT_TIME")
    
    return graph_documents

def main():
    setup_environment()
    
    # Load and preprocess emails
    emails = load_and_preprocess_emails("/data/taeho/graph_rag/cleaned_email_data.json")
    
    # Split emails
    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts, subjects, cc, froms, tos, dates, times = split_emails(emails[:0], text_splitter)
    
    # Create LLM and transformer
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    # Create graph documents
    graph_documents = create_graph_documents(texts, subjects, cc, froms, tos, dates, times, llm_transformer)
    
    # Create Neo4j graph and add documents
    graph = Neo4jGraph()
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    
    # Create vector index
    vector_index = Neo4jVector.from_existing_graph(
        UpstageEmbeddings(model="solar-embedding-1-large"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

if __name__ == "__main__":
    main()