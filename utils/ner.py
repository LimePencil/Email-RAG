from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field

import json
class Person(BaseModel):
    """Information about a person."""
    name: Optional[str] = Field(default = None, description="The name of the person. It's commonly comprised of three korean words.(e.g. 황태호, 장원준, 윤태호) Only Korean name is considered. If you encounter the words like '교수님', '님', '야' at the end, make sure that it is not part of the name.")

class People(BaseModel):
    person_list: list[Person]

class EmailAddress(BaseModel):
    address: Optional[str] = Field(default = None, description="The address of the email. Commonly, there should be @ in the middle and .com at the end. E.g. wj1234@gmail.com")

class EmailAddresses(BaseModel):
    address_list: list[EmailAddress]

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
def recognize_name(text: str) -> People:

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm in terms of person name. "
                "Only extract relevant information from the text. "
                "Please don't repeatedly extract duplicated names. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

    runnable = prompt | llm.with_structured_output(schema=People)
    try:
        result = runnable.invoke({"text": text})
    except Exception as e:
        print(e)
        print("text: ",text)
    result_modified = People(person_list=[])
    for person in result.person_list:
        if(person.name!=None):
            result_modified.person_list.append(person)
    return result_modified

def recognize_address(text: str) -> EmailAddresses:

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "Please don't repeatedly extract duplicated addresses."
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

    runnable = prompt | llm.with_structured_output(schema=EmailAddresses)
    try:
        result = runnable.invoke({"text": text})
    except Exception as e:
        print(e)
        print("text: ",text)
    result_modified = EmailAddresses(address_list=[])
    for address in result.address_list:
        if(address.address!=None):
            result_modified.address_list.append(address)
    return result_modified