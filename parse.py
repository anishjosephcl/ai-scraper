from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

template = (
    "You are tasked with extracting specific information from the following dom content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **Only Table Content:** Only extract the table content. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

model = GoogleGenerativeAI(model="gemini-pro",google_api_key="AIzaSyA7PUMgam6YHY0bxgDAWp6rTY3HIJpgrkQ")


def parse_with_gemini(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke(
            {"dom_content": chunk, "parse_description": parse_description}
        )
        print(f"Parsed batch: {i} of {len(dom_chunks)}")
        parsed_results.append(response)

    return "\n".join(parsed_results)