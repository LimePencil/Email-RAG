import mailparser
import os
import json
from attachment_parsing import parse_attachment

from bs4 import BeautifulSoup

NUMBER_OF_EMAILS = 20

folder_paths = ['C:/Users/jaeyo/Desktop/Coding/Upstage/email_files/']
for folder_path in folder_paths:
    
    # recent emails first
    file_names = os.listdir(folder_path)[::-1][:NUMBER_OF_EMAILS]
    json_data = []

    for n,file in enumerate(file_names):
        
        file_path = folder_path + file
        
        mail = mailparser.parse_from_file(file_path)
        
        # from html extract text
        # html: mail.mail["body"]
        # parse html
        soup = BeautifulSoup(mail.mail["body"], 'lxml')

        # Extract text from the HTML
        extracted_text = soup.get_text(separator='\n', strip=True)
        if extracted_text == "":
            print()
        # save the email data in a dictionary
        if mail.mail.get("cc") is None:
            mail.mail["cc"] = []
        else:
            cc = mail.mail["cc"]
        if mail.mail.get("subject") is None:
            mail.mail["subject"] = []
            continue
        if mail.mail.get("to") is None:
            mail.mail["to"] = []
        email_data = {
            'subject': mail.mail["subject"],
            'from': mail.mail["from"],
            'to': mail.mail["to"],
            'cc': mail.mail["cc"],
            'date': mail.date_json,
            'text_body': extracted_text,
            'attachments': parse_attachment(file_path, './attachments/')
        
        }
        json_data.append(email_data)
        
        if n%100 == 0:
            print(f'{n} emails parsed')
    
# save the email data in a json file
with open('email_data.json', 'w',encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)