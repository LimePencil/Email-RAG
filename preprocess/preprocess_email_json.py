import json
import pandas as pd
import re
import unicodedata

# Load the JSON file
with open('email_data.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

# Function to clean text
def clean_text(text):
    # Replace newlines, tabs, multiple spaces, and carriage returns
    text = re.sub(r'[\n\t\r]+', ' ', text)
    # Replace multiple exclamation marks with a period
    text = re.sub(r'!+', '.', text)
    # Remove any characters that are not alphanumeric or punctuation and korean
    text = re.sub(r'[^\w\s\.\?\!\,ㄱ-ㅎㅏ-ㅣ가-힣@:-_]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Trim leading and trailing spaces
    text = text.strip()
    return text
# mail_boundary
# Apply the cleaning function to each document
for d in data:
    if len(d["text_body"].split('mail_boundary')) == 1:
        print()
    else:
        d["text_body"] = d["text_body"].split('mail_boundary')[1]
    d["date"]
    d["text_body"]= unicodedata.normalize("NFKD", d["text_body"])
    d["text_body"] = clean_text(d["text_body"])


# save the cleaned data in a json file
with open('cleaned_email_data.json', 'w', encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
