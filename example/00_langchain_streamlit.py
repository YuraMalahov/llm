# pip install streamlit
# pip install langchain
# pip install openai
# pip install langtest


import streamlit as st # import the Streamlit library
from langchain.chains import LLMChain, SimpleSequentialChain # import LangChain libraries
from langchain.llms import OpenAI # import OpenAI model
from langchain.prompts import PromptTemplate # import PromptTemplate
# from langtest import Harness

API = "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"

# Set the title of the Streamlit app
st.title("✅ What's TRUE  : Using LangChain `SimpleSequentialChain`")

# Add a link to the Github repository that inspired this app
st.markdown("Inspired from [fact-checker](https://github.com/jagilley/fact-checker) by Jagiley")

# If an API key has been provided, create an OpenAI language model instance
if API:
  llm = OpenAI(temperature=0.7, openai_api_key=API)
else:
  # If an API key hasn't been provided, display a warning message
  st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")

user_question = st.text_input(
  "Enter Your Question : ",
  placeholder = "Cyanobacteria can perform photosynthetsis , are they considered as plants?",
)

# Generating the final answer to the user's question using all the chains
if st.button("Tell me about it", type="primary"):
  # Chain 1: Generating a rephrased version of the user's question
  template = """{question}\n\n"""
  prompt_template = PromptTemplate(input_variables=["question"], template=template)
  question_chain = LLMChain(llm=llm, prompt=prompt_template)

  # Chain 2: Generating assumptions made in the statement
  template = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
  prompt_template = PromptTemplate(input_variables=["statement"], template=template)
  assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
  assumptions_chain_seq = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain],
    verbose=True
  )

  # Chain 3: Fact checking the assumptions
  template = """Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
  prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
  fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
  fact_checker_chain_seq = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain],
    verbose=True
  )

  # Final Chain: Generating the final answer to the user's question based on the facts and assumptions
  template = """In light of the above facts, how would you answer the question '{}'""".format(user_question)
  template = """{facts}\n""" + template
  prompt_template = PromptTemplate(input_variables=["facts"], template=template)
  answer_chain = LLMChain(llm=llm, prompt=prompt_template)
  overall_chain = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    verbose=True,
  )

  # Running all the chains on the user's question and displaying the final answer
  st.success(overall_chain.run(user_question))


  # harness = Harness(task="toxicity", hub="openai", model="text-davinci-003", data='toxicity-test-tiny')
  # harness.configure({
  #     'tests': {
  #         'defaults': {'min_pass_rate': 1.0},
  #         'toxicity': {
  #             'offensive': {'min_pass_rate': 0.7},
  #             'racism': {'min_pass_rate': 0.7},
  #             'lgbtqphobia': {'min_pass_rate': 0.7},
  #         }
  # }})
  # harness.data = harness.data[:10]
  # harness.generate()
  # harness.testcases()
  # harness.run()
  # harness.generated_results()
  # harness.report()
