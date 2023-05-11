import pickle
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Build the LLM

# insert your OpenAI API key
openai_apikey = st.secrets["openai_apikey"]
os.environ["OPENAI_API_KEY"] = openai_apikey

# the location of the PFD file used to train your model
reader = PdfReader('data/FIFA_Football_Agent_Exam_Study_Materials.pdf')

# read the data from the PDF file and put it into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# split the text into smaller chunks so that during information retreival we don't hit the token size limit.
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 3500,
    chunk_overlap  = 600,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Create docsearch
docsearch = FAISS.from_texts(texts, embeddings)

# Create chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Save the model
with open("llm_model.pkl", "wb") as f:
    pickle.dump(chain, f)

with open("docsearch_model.pkl", "wb") as f:
    pickle.dump(docsearch, f)
