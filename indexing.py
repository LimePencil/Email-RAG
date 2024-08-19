import os
from typing import List
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator

# Constants
ELASTIC_CLOUD_ID = "************************************************************************************************************************"
ELASTIC_API_KEY = "**************************************"

# Metadata field information
metadata_field_info = [
    AttributeInfo(
        name="subject",
        description="The subject of the email",
        type="string",
    ),
    AttributeInfo(
        name="cc",
        description="The CC (carbon copy) recipients of the email",
        type="string",
    ),
    AttributeInfo(
        name="from",
        description="The sender of the email",
        type="string",
    ),
    AttributeInfo(
        name="to",
        description="The recipient of the email",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the email was sent",
        type="integer",
    ),
    AttributeInfo(
        name="month",
        description="The month the email was sent",
        type="integer",
    ),
    AttributeInfo(
        name="day",
        description="The day the email was sent",
        type="integer",
    ),
    AttributeInfo(
        name="hour",
        description="The hour the email was sent",
        type="integer",
    ),
    AttributeInfo(
        name="minute",
        description="The minute the email was sent",
        type="integer",
    ),
    AttributeInfo(
        name="second",
        description="The second the email was sent",
        type="integer",
    ),
]

document_content_description = "KAIST academic email data"

def create_vectorstore(documents: List[Document], index_name: str) -> ElasticsearchStore:
    """
    Create and return an ElasticsearchStore instance.
    """
    return ElasticsearchStore.from_documents(
        documents,
        UpstageEmbeddings(model="solar-embedding-1-large"),
        index_name=index_name,
        es_cloud_id=ELASTIC_CLOUD_ID,
        es_api_key=ELASTIC_API_KEY, 
    )

def create_retriever(vectorstore: ElasticsearchStore, llm, query_constructor) -> SelfQueryRetriever:
    """
    Create and return a SelfQueryRetriever instance.
    """
    return SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        structured_query_translator=ElasticsearchTranslator(),
        search_kwargs={"k": 10},
    )

def setup_indexing(documents: List[Document], llm, query_constructor) -> SelfQueryRetriever:
    """
    Set up the indexing and retrieval system.
    """
    vectorstore = create_vectorstore(documents, "elasticsearch-self-query_for_demo2")
    retriever = create_retriever(vectorstore, llm, query_constructor)
    return retriever

# Usage
# retriever = setup_indexing(mails, llm, query_constructor)