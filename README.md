# Smart Email

## Dependency
https://github.com/LimePencil/Email-RAG/blob/main/requirements.txt

## Setup
### 1. Download email
Locate your email json file to data/graph_rag/ directory
### 2. upload to mongodb
https://github.com/LimePencil/Email-RAG/blob/main/utils/db_upload.py
### 3. indexing
Upload embedding to elastic cloud

## Test
```shell
$ uvicorn main:app --reload
```

### Contributors
