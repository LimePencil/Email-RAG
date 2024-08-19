# Smart Email
Our project made answering ai using rag and solar llm. Thanks to powerful embedding of solar, our powerful rag extremely increased accuracy of our answers.  
Furthermore, we leveraged self-querying which provides high performance answering with meta-data.  
Moreover, we tried deidentification of personal informations in order to prevent them from leaking to llm server.

## Environment
python 3.11.2

## Dependency
https://github.com/LimePencil/Email-RAG/blob/main/requirements.txt

## Setup
### 1. install requirements
```shell
pip install -r requirements.txt
```

### 2. Download email
Locate your email json file to ```data/graph_rag/``` directory
### 3. upload to mongodb
https://github.com/LimePencil/Email-RAG/blob/main/utils/db_upload.py
### 4. indexing
Upload embedding to elastic cloud  
https://github.com/LimePencil/Email-RAG/blob/main/utils/indexing_deidentification.py

## Test
```shell
$ uvicorn main:app --reload
```

### Contributors
