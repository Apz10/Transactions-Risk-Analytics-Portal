import pandas as pd
import json
import os

# Read the CSV file
df = pd.read_csv("C://Users//apoor//projects//Transactions-Risk-Analytics-Portal//test//data//transactions_generated_scenarios_gpt5.csv")

# Prepare JSONL file for OpenAI fine-tuning
jsonl_path = "C://Users//apoor//projects//Transactions-Risk-Analytics-Portal//test//data//transactions_generated_scenarios_gpt5.jsonl"
system_prompt = "You are a financial compliance assistant. Classify each transaction into one of the predefined scenarios."

with open(jsonl_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        user_content = json.dumps({
            "date": row["Date"],
            "txnid": row["TxnID"],
            "type": row["Type"],
            "amountEUR": row["AmountEUR"],
            "party": row["Party"],
            "counterparty": row["CounterpartyName"],
            "description": row["Description"],
            "location": row["Location"]
        }, ensure_ascii=False)
        assistant_content = row["Scenario"].replace("  ", " – ").replace("  ", " – ")
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Saved: {jsonl_path}")
