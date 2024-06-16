import os
import sys
import fitz  # PyMuPDF
import nltk
import numpy as np

import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

def tokenize_text(text):
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)

def embed_text(text):
    from langchain_community.embeddings import OpenAIEmbeddings
    embedding_model = OpenAIEmbeddings()
    return embedding_model.embed_query(text)

def preprocess_and_embed_data(data_dir, output_dir):
    all_embeddings = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            text = pdf_to_text(pdf_path)
            cleaned_text = clean_text(text)
            tokenized_text = tokenize_text(cleaned_text)
            embeddings = embed_text(' '.join(tokenized_text))
            all_embeddings.append(embeddings)

    all_embeddings = np.array(all_embeddings)
    np.save(os.path.join(output_dir, "healthcare_embeddings.npy"), all_embeddings)

# Directory containing the raw data files
data_dir = "data/indian_healthcare_data"
output_dir = "data/processed_data"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Preprocess and embed the data
preprocess_and_embed_data(data_dir, output_dir)

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = DirectoryLoader(data_dir)
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None
