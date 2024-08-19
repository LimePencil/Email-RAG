import json
import time
import mailparser
import os
import requests

def parse_attachment(email_file,path):
    
    mail = mailparser.parse_from_file(email_file)
    mail.write_attachments(path)
     
    api_key = "up_Ul3PcVtHuo1T9xwWhePcU062Rb2KW"
    
    
    url = "https://api.upstage.ai/v1/document-ai/layout-analysis"
    
    attachments = []
    
    for attachment in os.listdir(path):
        # use solar api to extract text from the attachment
        filename = path + attachment
        

        headers = {"Authorization": f"Bearer {api_key}"}
        with open(filename, "rb") as f:
            files = {"document": f}
            response = requests.post(url, headers=headers, files=files,data={"ocr":True})
            response_json = response.json()
            response_text=""
            response_html=[]
            for e in response_json["elements"]:            
                if "text" in e:
                    response_text += e["text"] + "\n"
                if "html" in e:
                    response_html.append(e["html"])
            # wait 1 second to avoid rate limit
            
            time.sleep(1)
    
    
    # remove the attachments after extracting text
    for attachment in os.listdir(path):
        os.remove(path + attachment)
    
    
    return response_text,response_html

if __name__ == "__main__":
    # test the function
    email_file = "C:/Users/jaeyo/Desktop/Coding/Upstage/email_files/20221024_143221_김예선(yeseon@kaist.ac.kr)_2022 글로벌 스타트업 인턴십 잡페어 (Global Startup Inter...eml"
    path = "C:/Users/jaeyo/Desktop/Coding/Upstage/attachments/"
    attachments = parse_attachment(email_file,path)
    print(attachments)