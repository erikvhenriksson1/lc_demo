# Bring in deps
from dotenv import load_dotenv

import os

import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

load_dotenv()
apikey = os.getenv("apikey")

os.environ['OPENAI_API_KEY']=apikey


st.title("Marketing email text generator")

product_prompt = st.text_input('Product context here')


user_info_prompt = st.text_input('User info prompt here')


#Prompt templates
email_subject_template = PromptTemplate(
    input_variables = ['product_prompt'],
    template = 'Write a catchy email subject for an email marketing this product: {product_prompt}'
)


email_body_template = PromptTemplate(
    input_variables = ['product_context','email_subject', 'wikipedia_research','user_info_prompt'],
    template = 'Write an email, marketing this product {product_context} with subject {email_subject}, ' +
                'to a customer with the following characteristics {user_info_prompt} while leveraging ' +
                'this wikipedia reasearch {wikipedia_research}'
)

#Memeory
subject_memory = ConversationBufferMemory(input_key='product_prompt',memory_key='subject_history')
text_memory = ConversationBufferMemory(input_key='user_info_prompt',memory_key='text_history')

### LLMs
llm = OpenAI(temperature=0.9) #creativity measure?

subject_chain = LLMChain(llm=llm,prompt=email_subject_template, verbose=True, output_key="email_subject", memory=subject_memory)
body_chain = LLMChain(llm=llm,prompt=email_body_template, verbose=True, output_key='email_body',memory=text_memory)

wiki = WikipediaAPIWrapper()


if product_prompt:
    # response = sequential_chain({'topic':prompt})

    subject = subject_chain.run(product_prompt)
    
    st.write(subject) 

    wiki_research = wiki.run(product_prompt)


    if user_info_prompt:

        body = body_chain.run(product_context = product_prompt, 
                              email_subject = subject, 
                              user_info_prompt = user_info_prompt,
                              wikipedia_research = wiki_research)

        st.write(body) 


    with st.expander('Subject History'):
        st.info(subject_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)

    if user_info_prompt:
        with st.expander('Body History'):
            st.info(text_memory.buffer)