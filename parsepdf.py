from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

template = (
    "You are tasked with extracting specific information from the following text content: {text_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information and convert it into csv format "
    "2. **Input format:** Each line has 3 columns separated by space Take this into account while parsing the data"
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

model = GoogleGenerativeAI(model="gemini-pro",google_api_key="AIzaSyA7PUMgam6YHY0bxgDAWp6rTY3HIJpgrkQ")


def parse_with_gemini_pdf(text_chunks):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_results = []

    for i, chunk in enumerate(text_chunks, start=1):
        response = chain.invoke(
            {"text_content": chunk}
        )
        print(f"Parsed batch: {i} of {len(text_chunks)}")
        parsed_results.append(response)
    df = pd.DataFrame(parsed_results)
    return df