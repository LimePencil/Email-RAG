import json

# open the json file
with open('email_data_cleaned.json', 'r',encoding="utf-8") as f:
    email_data = json.load(f)
    
for n,e in enumerate(email_data):
    e["uid"] = n

# save the email data in a json file
with open('email_data_cleaned.json', 'w',encoding="utf-8") as f:
    json.dump(email_data, f, ensure_ascii=False, indent=4)
        