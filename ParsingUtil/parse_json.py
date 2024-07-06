import json

# open the json file
with open('email_data.json', 'r',encoding="utf-8") as f:
    email_data = json.load(f)

parsed_emails = []
for e in email_data:
    parsed_emails.append(f"Subject: {e['subject']}\nFrom: {", ".join(e['from'])}\nTo: {", ".join(e['to'])}\nCC: {", ".join(e['cc'])}\nDate: {e['date']}\nEmail Body: {e['text_body']}\n\n")

# save the parsed emails in a json file
with open('parsed_emails.json', 'w',encoding="utf-8") as f:
    json.dump(parsed_emails, f, ensure_ascii=False, indent=4)