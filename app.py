"""Streamlit interface for the Data Cleaning Agent."""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from data_cleaning_agent import LightweightDataCleaningAgent
from data_cleaning_agent.utils import analyze_data_quality, build_cleaning_instructions

load_dotenv()

st.title("Data Cleaning Agent")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Reset cleaning results when a new file is uploaded
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_file_key") != file_key:
        st.session_state["_file_key"] = file_key
        st.session_state.pop("df_cleaned", None)

    df_raw = pd.read_csv(uploaded_file)

    st.caption(f"{df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head(), use_container_width=True)

    st.divider()

    # --- Data quality analysis ---
    issues = analyze_data_quality(df_raw)
    decisions: dict = {}

    if not issues:
        st.info("No missing values or IQR outliers detected. Cleaning will remove duplicates only.")
    else:
        st.subheader("Data Quality Issues")

        # Summary table
        summary_rows = []
        for issue in issues:
            summary_rows.append({
                "Column": issue["column"],
                "Type": issue["dtype"],
                "Missing": f"{issue['missing_count']} ({issue['missing_pct']}%)" if "missing_count" in issue else "—",
                "IQR Outliers": f"{issue['outlier_count']} ({issue['outlier_pct']}%)" if "outlier_count" in issue else "—",
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

        st.subheader("Cleaning Decisions")

        # Column header row
        h1, h2, h3 = st.columns([2, 2, 2])
        h1.caption("Column")
        h2.caption("Missing values")
        h3.caption("IQR Outliers (numeric only)")

        for issue in issues:
            col = issue["column"]
            dtype = issue["dtype"]
            decisions[col] = {}

            c1, c2, c3 = st.columns([2, 2, 2])
            c1.markdown(f"**{col}**  \n`{dtype}`")

            if "missing_count" in issue:
                opts = (
                    ["impute with mean", "impute with median", "drop rows", "custom"]
                    if dtype == "numeric"
                    else ["impute with mode", "drop rows", "custom"]
                )
                missing_choice = c2.selectbox(
                    f"missing_{col}",
                    opts,
                    key=f"missing_{col}",
                    label_visibility="collapsed",
                )
                if missing_choice == "custom":
                    custom = c2.text_input(
                        "Custom instruction",
                        key=f"missing_custom_{col}",
                        placeholder="e.g. fill with 0",
                    )
                    decisions[col]["missing"] = custom or "custom (no details provided)"
                else:
                    decisions[col]["missing"] = missing_choice
            else:
                c2.caption("—")

            if "outlier_count" in issue:
                outlier_choice = c3.selectbox(
                    f"outliers_{col}",
                    ["replace with mean", "replace with median", "drop rows", "custom"],
                    key=f"outliers_{col}",
                    label_visibility="collapsed",
                )
                if outlier_choice == "custom":
                    custom = c3.text_input(
                        "Custom instruction",
                        key=f"outliers_custom_{col}",
                        placeholder="e.g. cap at 99th percentile",
                    )
                    decisions[col]["outliers"] = custom or "custom (no details provided)"
                else:
                    decisions[col]["outliers"] = outlier_choice
            else:
                c3.caption("—")

    st.divider()

    # --- Clean button ---
    if st.button("Clean Data", type="primary"):
        instructions = build_cleaning_instructions(decisions)
        with st.spinner("Cleaning..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            agent.invoke_agent(data_raw=df_raw, user_instructions=instructions)
            st.session_state["df_cleaned"] = agent.get_data_cleaned()

    # --- Results ---
    if "df_cleaned" in st.session_state:
        df_cleaned = st.session_state["df_cleaned"]
        st.success("Done!")
        st.subheader("Cleaned Data")
        st.caption(f"{df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
        st.dataframe(df_cleaned.head(), use_container_width=True)
        st.download_button(
            "Download Cleaned Data",
            data=df_cleaned.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
