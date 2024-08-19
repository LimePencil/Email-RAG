# Smart Email
<<<<<<< HEAD
<<<<<<< HEAD
Our project made answering ai using rag and solar llm. Thanks to powerful embedding of solar, our powerful rag extremely increased accuracy of our answers. 
Furthermore, we leveraged self-querying which provides high performance answering with meta-data. 
=======
Our project made answering ai using rag and solar llm. Thanks to powerful embedding of solar, our powerful rag extremely increased accuracy of our answers.  
Furthermore, we leveraged self-querying which provides high performance answering with meta-data.  
>>>>>>> 1b6bf9e5b4125598c1279ab95ae37a58029d9ef7
=======
Our project made answering ai using rag and solar llm. Thanks to powerful embedding of solar, our powerful rag extremely increased accuracy of our answers.  
Furthermore, we leveraged self-querying which provides high performance answering with meta-data.  
>>>>>>> b7a587898b33b390b2bd97b7bf888e870ce6c918
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
### 3. upload email to mongodb
https://github.com/LimePencil/Email-RAG/blob/main/utils/db_upload.py
### 4. indexing
Upload embedding to elastic cloud  
https://github.com/LimePencil/Email-RAG/blob/main/utils/indexing_deidentification.py

## Test
```shell
$ uvicorn main:app --reload
```

### Contributors
