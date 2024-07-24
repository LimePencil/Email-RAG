import json

# open the json file

with open('email_data.json', 'r',encoding="utf-8") as f:
    email_data = json.load(f)
    for e in email_data:
        print(e["subject"])