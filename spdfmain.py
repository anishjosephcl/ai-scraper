import streamlit as st

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from parsepdf import parse_with_gemini_pdf

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents

#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 200)
    text_chunks = text_splitter.split_documents(extracted_data)
    ##print(text_chunks[0].page_content)
    return text_chunks

# Streamlit UI
st.title("AI PDF Scraper")
url = st.text_input("Enter directory name")

# Step 1: Scrape the Website
if st.button("Scrape PDF"):
    if url:
        st.write("Scraping the pdf...")

        doc = load_pdf(url)
        text = text_split(doc)
        parsed_result = parse_with_gemini_pdf(text)
        st.write(parsed_result)

