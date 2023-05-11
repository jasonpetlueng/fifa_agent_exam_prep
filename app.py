import requests
import streamlit as st
from streamlit_lottie import st_lottie
import pickle
import os
from langchain.llms import OpenAI
import openai

# insert your OpenAI API key
openai.api_key = openai_apikey

# Load the model
with open("llm_model.pkl", "rb") as f:
    chain = pickle.load(f)

with open("docsearch_model.pkl", "rb") as f:
    docsearch = pickle.load(f)

#Create the streamlit app

st.set_page_config(page_title="FIFA Agent Exam Prep", page_icon=":soccer_ball:", layout="wide")

#function to load the lottie agent_image
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#use local CSS file to set the style
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

#load agent image
agent_image = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_5r1heU.json")

#header section
with st.container():
    st.title("FIFA Agent Exam Prep")
    st.subheader("An AI assistant to help you prepare for the FIFA agent exam")
    st.write("Did you know that more than half of the people who take the FIFA agent exam fail? Rather than pour hundred of pages of dense regulations, let this AI assistant powered by GPT answer your questions, help you study, and guide you along your journey to acing the FIFA agent exam!")
    st.write("##")
    st.write("Created by Jason Petlueng")
    st.write("[Jason's LinkedIn Profile](https://www.linkedin.com/in/jasonpetlueng/)")

#the model section -- text box and button
st.write("---")
with st.form("string_input_form"):
    question = st.text_input(
        "Enter Your Question: ",
        key="question",
        placeholder="What is the maximum fee a football agent can receive?",
    )
    submit_button = st.form_submit_button(label="Ask")

if submit_button or st.session_state.get("question_changed"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner(text="Loading..."):
            docs = docsearch.similarity_search(question)
            answer = chain.run(input_documents=docs, question=question)
            st.write("Answer:", answer)
            st.session_state["question_changed"] = True

#what/how section
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What is FIFA Agent Exam Prep?")
        st.write("FIFA Agent Exam Prep is an artificial intelligence tool powered by Generative Pre-trained Transformer (GPT). Trained on FIFA’s latest statutes, rules, and regulations, this large language model (LLM) allows users to use natural language to easily access information needed to pass the FIFA Agent Exam.")
        st.header("How can I use FIFA Agent Exam Prep to help me prepare for the exam?")
        st.write("Think of FIFA Agent Exam Prep as an experienced agent who knows all the ins and out of every rule and regulation regarding player representation. Treat FIFA Agent Exam Prep as a mentor who you can ask questions of to help you understand what rules an agent must follow and what an agent can and can’t do. Not sure if a certain clause is allowed to be included in a contract? Forgot what the fee cap for a certain service is? Ask FIFA Agent Exam Prep!")
    with right_column:
        st_lottie(agent_image, height=430)

#contact form
with st.container():
    st.write("---")
    st.header("Get in touch with Jason")

    contact_form = """
    <form action="https://formsubmit.co/jason.petlueng@gmail.com" method="POST">
        <input type="text" name="name" placeholder="Your Name" required>
        <input type="email" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Type Your Message Here" required></textarea>
        <button type="submit">Send</button>
     </form>
     """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

st.write("##")
st.write("Disclaimer: FIFA Agent Exam Prep is an independent tool and is not affiliated with or endorsed by FIFA.")
