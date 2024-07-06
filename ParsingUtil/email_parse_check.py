import mailparser
import os
import json

    
# for all emails in a file

folder_path = 'email_files/'
# get all files in the folder
file_names = os.listdir(folder_path)
json_data = []

for n,file in enumerate(file_names):
    
    file_path = folder_path + file
    
    mail = mailparser.parse_from_file(file_path)
    # save the email data in a dictionary
    email_data = {
        'subject': mail.subject,
        'from': mail.from_,
        'to': mail.to,
        'cc': mail.cc,
        'date': mail.date_json,
        'text_body': mail.text_plain,
    
    }
    json_data.append(email_data)
    
    if n%100 == 0:
        print(f'{n} emails parsed')
    
# save the email data in a json file
with open('email_data.json', 'w',encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
    