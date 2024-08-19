
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from tqdm import tqdm
from langchain_chroma import Chroma
import os
import warnings




import json

# load json
# /data/taeho/self-rag/email_data.json
with open("/data/taeho/Email-RAG/cleaned_email_data.json") as f:
    data = json.load(f)

from openai import OpenAI
os.environ["OPENAI_API_KEY"]  =  os.environ.get('OPENAI_API_KEY')
           
client = OpenAI(
  organization='org-by1AcXRM6jsKm96NkpC6l7Yt',
)


# make result json
result = []
# random shuffle
import random
random.shuffle(data)
cnt = 0
for i, d in tqdm(enumerate(data)):
    if cnt > 300:
        break
    try:
        # question generation
        completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": '''output form:
{
"query": "", 
"answer": "",
}
______
given e-mail, make question and answer (longform form) in korean. You should know that you can get multiple emails a day. output must be json form. Do not any explanation.'''},
                            {"role": "user", "content": str(d)}]
                        ,
                        temperature=1,
                        top_p=1,
                        n=1
        )
        response = completion.choices[0].message.content

        # parse json to dict
        response = json.loads(response)
        questions = response["query"]
        answers = response["answer"]


        temp = {
            "email_id": d["uid"],
            "questions": questions,
            "answers": answers
        }
        result.append(temp)
        cnt += 1
        if i % 100 == 0:
            with open("/data/taeho/email_questions_longform.json", "w") as f:
                json.dump(result, f)
    except Exception as e:
        print(e)
        print(i)
        continue

# dump json
with open("/data/taeho/email_questions_longform.json", "w") as f:
    json.dump(result, f)

