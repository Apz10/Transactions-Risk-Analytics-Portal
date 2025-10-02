import streamlit as st
import pandas as pd
import model_run.counterparty_analysis as cp_analysis_agent 
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from PIL import Image

# Sidebar navigation
st.sidebar.title("Transactions Risk Analytics Portal")
page = st.sidebar.radio("Go to", ["Welcome", "Counter Party Analysis", "Data Counter Party Analysis", "Training Data Generation"])

# Welcome Page
if page == "Welcome":
	st.title("Transactions Risk Analytics")
	st.markdown("""
				Welcome to the Transactions Risk Analytics platform. Use the sidebar to navigate.

				**We label transactions with following scenarios under Counter Party Analytics:**

					- Business Income – Export
					- Business Income – Domestic
					- Business Expense – Supplier
					- Regulatory Expense
					- Private Investment Company – Related Party
					- Improper – Possible Tax Reduction
					- Recurring Client Sales
					- Unusual – Third Party
			 Finally, we provide insights into potential risks and anomalies in transaction data.
				""")
	st.markdown("Trained on small dataset, large training in-progress :)")
	ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
	# # # st.write(ROOT_DIR)
	# # bs = (ROOT_DIR +"/webresource/images/TrainingGraph.png").replace("/", "\\")
	# # # st.write(bs)
	# st.image("./TrainingGraph.png", caption="", use_container_width =True)


# Counter Party Analysis Page
elif page == "Counter Party Analysis":
	st.title("Counter Party Analysis Portal")

	# Step3a: Upload CSV
	uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
	llm_agent = st.selectbox("Choose LLM Agent", ["gpt-3.5-turbo", "gpt-4", "ft:gpt-4.1-nano-2025-04-14:personal::CLygEFH0"])

	if uploaded_file:
		# Step3e: Validate CSV
		try:
			df = pd.read_csv(uploaded_file)
			st.success("CSV file uploaded successfully.")
		except Exception as e:
			st.error(f"Error reading CSV: {e}")
			st.stop()

		# Step3b: Choose LLM agent
		

		# Step3c: Edit CSV
		edited_df = st.data_editor(df, num_rows="dynamic").head(10)

		# Step3d: Submit CSV
	if st.button("Submit CSV for Risk Analysis"):
		# Step3f: Send to OpenAI model
		try:

			prompt = f'''Label all the transaction data with one of the scenario Business Income Export,Business Income Domestic,Business Expense Supplier,Regulatory Expense,Private Investment Company Related Party,Improper Possible Tax Reduction,Recurring Client Sales,Unusual Third Party, only output the scenario and transaction id.
			{edited_df.to_csv(index=False)}
			'''
			if llm_agent.startswith("ft:"):
				st.subheader("Fine-tuned Model Output")
				aggregated_results = edited_df.groupby('CounterpartyName')['AmountEUR'].sum().reset_index()
				counterparty_list = edited_df['TxnID'].tolist()
				result = cp_analysis_agent.run_counterparty_analysis(prompt, "gpt-4")
				sus_cp = []
				for i in counterparty_list:
					if i in result:
						sus_cp.append(i)
				st.write("Suspicious Counterparties to look into:")
				sus_cp_df = pd.DataFrame(sus_cp, columns=["Suspicious_TxnId"])
				result_sus = pd.merge(edited_df, sus_cp_df, left_on='TxnID', right_on='Suspicious_TxnId', how='inner')
				st.write(result_sus)
				st.write(result)
			else:
				result = cp_analysis_agent.run_counterparty_analysis(prompt, llm_agent)
				st.subheader("Model Output")
				# Step3g: Display result
				st.write(result)
		except Exception as e:
			st.error(f"No file submitted")
			st.markdown("Error from Agent Run:")
			st.error(f"Error from OpenAI: {e}")

# Data Counter Party Analysis
elif page == "Data Counter Party Analysis":
	st.title("Data For Counter Party Analysis")

	st.markdown("Sample Test Data for Counter Party Analysis (by Mani):")
	ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

	df = pd.read_csv(os.path.join(ROOT_DIR, "test/data/transactions.csv"))
	edited_df = st.data_editor(df, num_rows="dynamic")


	st.markdown("Training Data Generated for Counter Party Analysis (by GPT-5):")
	df = pd.read_csv(os.path.join(ROOT_DIR, "test/data/transactions_generated_scenarios_gpt5.csv"))
	edited_df = st.data_editor(df, num_rows="dynamic")

# Data Counter Party Analysis
elif page == "Training Data Generation":
	st.title("Training Data For Counter Party Analysis")

	st.markdown("Training JSONL Data for Counter Party Analysis:")
	ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

	with open(os.path.join(ROOT_DIR, "test/data/transactions_finetune.jsonl"), "r", encoding="utf-8") as f:
		for line in f:
			record = json.loads(line)
			print(record)
	st.write(record)

	st.markdown("Validation JSONL Data for Counter Party Analysis:")
	with open(os.path.join(ROOT_DIR, "test/data/transactions_training.jsonl"), "r", encoding="utf-8") as f:
		for line in f:
			record = json.loads(line)
			print(record)
	st.write(record)

	st.markdown("Validation JSONL Data for Counter Party Analysis:")
	with open(os.path.join(ROOT_DIR, "test/data/transactions_validation.jsonl"), "r", encoding="utf-8") as f:
		for line in f:
			record = json.loads(line)
			print(record)
	st.write(record)

st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 40px;
        right: 30px;
        opacity: 1;
        font-size: 20px;
        color: #888;
        z-index: 9999;
        pointer-events: none;
        user-select: none;
    }
    </style>
    <div class="watermark">v.1.0.4</div>
    """,
    unsafe_allow_html=True
)