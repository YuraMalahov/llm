from langchain.llms import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

API = "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"

llm = OpenAI(temperature=0.5, openai_api_key=API)
chat_model = ChatOpenAI(temperature=0.5, openai_api_key=API)

# *********************************************
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

text = "What would be a good company name for a company that makes colorful socks?"

r1 = llm.predict(text)
print(r1)

r2 = chat_model.predict(text)
print(r2)

# # *********************************************
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

messages = [HumanMessage(content=text)]

r1 = llm.predict_messages(messages)
print(r1.content)

r2 = chat_model.predict_messages(messages)
print(r2.content)

# *********************************************
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
message = prompt.format(product="colorful socks")

r1 = llm.predict(message)
print(r1)

# *********************************************
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
message = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")

r1 = llm.predict_messages(message)
print(r1.content)
