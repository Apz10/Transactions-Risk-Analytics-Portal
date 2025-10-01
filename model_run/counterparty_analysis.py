import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# If using langraph, import relevant modules (example below)
# from langraph import Graph

def run_counterparty_analysis(prompt: str, llm_agent: str) -> str:
    # Step 1: Initialize OpenAI model
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=llm_agent)

    # Step 2: Read CSV from prompt (assuming prompt contains CSV data)
    # Here, we extract the CSV part from the prompt for demonstration purposes.
    # In practice, you might want to pass the DataFrame directly.
    csv_data = prompt.split("\n", 1)[1]  # Get everything after the first line
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))

    # Step 3: Send CSV with prompt
    prompt_template = PromptTemplate(
        input_variables=["transactions"],
        template="Detect suspicious activity from the transactions below:\n{transactions}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Prepare transactions as string (limit rows if needed)
    transactions_str = df.to_csv(index=False)

    # Step 4: Get result and return
    result = chain.run({"transactions": transactions_str})
    return result
