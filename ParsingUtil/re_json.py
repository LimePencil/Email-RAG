import json

# open the json file
with open('email_data.json', 'r',encoding="utf-8") as f:
    email_data = json.load(f)

# use regex to replace all emojis and newline in the email body with a space
import re
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

for e in email_data:
    e["text_body"] = emoji_pattern.sub(r'', "".join(e["text_body"]))
    e["text_body"] = "".join(e["text_body"]).replace('\n', ' ')
    # also replace line separators and paragraph separators
    e["text_body"] = "".join(e["text_body"]).replace('\u2028', ' ')
    e["text_body"] = "".join(e["text_body"]).replace('\u2029', ' ')
    # &nbsp; is a non-breaking space in HTML replace it to a space
    e["text_body"] = "".join(e["text_body"]).replace('&nbsp;', ' ')
    
    e["subject"] = emoji_pattern.sub(r'', "".join(e["subject"]))
    e["subject"] = "".join(e["subject"]).replace('\n', ' ')
    e["subject"] = "".join(e["subject"]).replace('\u2028', ' ')
    e["subject"] = "".join(e["subject"]).replace('\u2029', ' ')
    e["subject"] = "".join(e["subject"]).replace('&nbsp;', ' ')
    
# save the email data in a json file
with open('email_data_cleaned.json', 'w',encoding="utf-8") as f:
    json.dump(email_data, f, ensure_ascii=False, indent=4)