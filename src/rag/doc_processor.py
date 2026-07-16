from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("data/pdf1.pdf")
docs = loader.load()


def load_pdf(path:str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return  docs

# Quebrar o documento em pedaços menores
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

