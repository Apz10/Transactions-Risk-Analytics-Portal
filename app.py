import streamlit as st
import pandas as pd
import model_run.counterparty_analysis as cp_analysis_agent 
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Sidebar navigation
st.sidebar.title("Transactions Risk Analytics Portal")
page = st.sidebar.radio("Go to", ["Welcome", "Counter Party Analysis", "Data Counter Party Analysis"])

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
				""")


# Counter Party Analysis Page
elif page == "Counter Party Analysis":
	st.title("Counter Party Analysis Portal")

	# Step3a: Upload CSV
	uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
	llm_agent = st.selectbox("Choose LLM Agent", ["gpt-3.5-turbo", "gpt-4", "gpt-4.1-nano-2025-04-14"])

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
			prompt = "Detect suspicious transactions in the following data:\n" + edited_df.to_csv(index=False)

			result = cp_analysis_agent.run_counterparty_analysis(prompt, llm_agent)
			# Step3g: Display result
			st.subheader("Model Output")
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
    <div class="watermark">v.1.0.1</div>
    """,
    unsafe_allow_html=True
)