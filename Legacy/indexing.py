import warnings

warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage

from langchain_chroma import Chroma
import os


import json

# load json
# /data/taeho/self-rag/email_data.json
with open("/data/taeho/self-rag/email_data_cleaned.json") as f:
    data = json.load(f)

from langchain_upstage import UpstageEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
# to string
docs = [{"id": d["uid"], "text": d["text_body"]} for d in data if len(d["text_body"]) > 0]


llm = ChatUpstage()

prompt_template = PromptTemplate.from_template(
    """
    Please provide most correct answer from the following context. 
    If the answer is not present in the context, please write "The information is not present in the context."
    ---
    Question: {question}
    ---
    Context: {Context}
    """
)




from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0")
tokenizer.pad_token = tokenizer.eos_token


# docs = text_splitter.create_documents(docs)
for i in range(len(docs)):
    a = tokenizer.encode(docs[i]["text"], return_tensors="pt", max_length=4000, truncation=True)
    docs[i]["text"]= tokenizer.decode(a[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    

sample_docs = []
for i in range(len(docs)):
    if docs[i]["text"].strip() == "":
        continue
    froms = ""
    for f in data[docs[i]["id"]]["from"]:
        froms += f'{f[0]} ({f[1]})'
        froms += ", "
    froms = froms.strip()

    tos = ""
    for f in data[docs[i]["id"]]["to"]:
        tos += f'{f[0]} ({f[1]})'
        tos += ", "
    tos = tos.strip()

    date = str(data[docs[i]["id"]]["date"])[1:-10]
    
    ccs = ""
    for f in data[docs[i]["id"]]["cc"]:
        ccs += f'{f[0]} ({f[1]})'
        ccs += ", "
    ccs = ccs.strip()
    
    sample_docs.append(Document(page_content=docs[i]["text"], metadata={"subject": str(data[docs[i]["id"]]["subject"]), "from": froms, "to": tos, "date": date, "cc" : ccs, "uid" : str(docs[i]["id"])}))
    # sample_docs.append(Document(page_content=docs[i]["text"], metadata={"s" : "hello"}))



sample_docs = sample_docs[-1000:]


# if ./chroma_db not exist
if not os.path.exists("/data/taeho/chroma_final_2"):
    vectorstore = Chroma.from_documents(
            documents=sample_docs,
            embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
            persist_directory="/data/taeho/chroma_final_2"
    )
else:
    vectorstore = Chroma(persist_directory="/data/taeho/chroma_final_2",
                embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"))
    
index_count = vectorstore._collection.count()
print(f"Vector store index count: {index_count}")

from langchain_upstage import UpstageGroundednessCheck

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(
        name="subject",
        description="The subject of the email",
        type="string",
    ),
    AttributeInfo(
        name="from",
        description="who sent the email",
        type="string",
    ),
    AttributeInfo(
        name="to",
        description="who received the email",
        type="string",
    ),
    AttributeInfo(
        name="date", description="", type="date"
    ),
]


document_content_description = "The content of the email"

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

groundedness_check = UpstageGroundednessCheck()

retriever = vectorstore.as_retriever()
chain = prompt_template | llm | StrOutputParser()
while True:
    query = input("질문을 입력해주세요: ")
    docs = retriever.invoke(query)
    for _ in range(5):
        answer = chain.invoke({"question": query, "Context": docs})
        gc_result = groundedness_check.invoke({"context": docs, "answer": answer})
        print("GC check result: ", gc_result)
        if gc_result.lower().startswith("grounded"):
            print("✅ Groundedness check passed")
            break
        else:
            print("❌ Groundedness check failed")
            continue
    print("Answer: ", answer)
    