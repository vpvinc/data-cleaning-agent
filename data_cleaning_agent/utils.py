# Utility functions for lightweight data cleaning agent

import re
import logging
from typing import Dict, List, Optional

import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""
    
    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if python_code_match:
            return python_code_match.group(1).strip()
        return text


def get_dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a simple summary of a DataFrame for the LLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    
    Returns
    -------
    str
        A text summary of the DataFrame.
    """
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])
    
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
    
    summary = f"""
        Dataset Summary:
        ----------------
        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}"""

    return summary.strip()


def analyze_data_quality(df: pd.DataFrame) -> List[dict]:
    """
    Analyze a DataFrame for missing values and IQR outliers.

    Only returns columns that have at least one issue. Outlier detection
    is only applied to numeric columns using the 1.5 × IQR rule.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    List[dict]
        One dict per affected column with keys:
        - column (str)
        - dtype ('numeric' | 'categorical')
        - missing_count, missing_pct  (only when missing values exist)
        - outlier_count, outlier_pct  (only when outliers exist, numeric only)
    """
    issues = []
    n = len(df)
    if n == 0:
        return issues

    for col in df.columns:
        info: dict = {"column": col}
        missing = int(df[col].isna().sum())

        if pd.api.types.is_numeric_dtype(df[col]):
            info["dtype"] = "numeric"
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            outliers = int(outlier_mask.sum())
        else:
            info["dtype"] = "categorical"
            outliers = 0

        if missing > 0:
            info["missing_count"] = missing
            info["missing_pct"] = round(missing / n * 100, 1)

        if outliers > 0:
            info["outlier_count"] = outliers
            info["outlier_pct"] = round(outliers / n * 100, 1)

        if missing > 0 or outliers > 0:
            issues.append(info)

    return issues


def build_cleaning_instructions(decisions: Dict[str, dict]) -> Optional[str]:
    """
    Convert per-column user decisions into a text instruction string for the LLM.

    Parameters
    ----------
    decisions : dict
        Keys are column names. Values are dicts with optional keys:
        - 'missing'  : one of 'impute with mean', 'impute with median',
                       'impute with mode', 'drop rows'
        - 'outliers' : one of 'replace with mean', 'replace with median',
                       'drop rows'

    Returns
    -------
    str or None
        Formatted instruction string, or None if no decisions were made.
    """
    lines = []
    for col, actions in decisions.items():
        parts = []
        if actions.get("missing"):
            parts.append(f"missing values → {actions['missing']}")
        if actions.get("outliers"):
            parts.append(f"IQR outliers → {actions['outliers']}")
        if parts:
            lines.append(f"- Column '{col}': {'; '.join(parts)}")

    if not lines:
        return None

    return (
        "Apply the following column-specific cleaning rules:\n"
        + "\n".join(lines)
        + "\nAlso remove duplicate rows."
    )


def execute_agent_code(state, data_key, code_snippet_key, result_key, error_key, agent_function_name):
    """
    Execute the generated agent code on the data.
    
    Parameters
    ----------
    state : dict
        The current state containing data and code.
    data_key : str
        Key in state where the input data is stored.
    code_snippet_key : str
        Key in state where the generated code is stored.
    result_key : str
        Key to store the result in.
    error_key : str
        Key to store any error message in.
    agent_function_name : str
        Name of the function to execute from the generated code.
    
    Returns
    -------
    dict
        Dictionary with result and error keys.
    """
    logger.info("Executing agent code")
    
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    df = pd.DataFrame.from_dict(data)
    
    # Execute the LLM-generated code in isolated namespace
    # Note: exec() can be risky - only use with trusted LLM-generated code
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    # Get the function from executed code
    agent_function = local_vars.get(agent_function_name)
    if not agent_function or not callable(agent_function):
        raise ValueError(f"Function '{agent_function_name}' not found in generated code.")
    
    # Run the function and handle errors
    agent_error = None
    result = None
    try:
        result = agent_function(df)
        if isinstance(result, pd.DataFrame):
            result = result.to_dict()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        agent_error = f"An error occurred during data cleaning: {str(e)}"
    
    return {result_key: result, error_key: agent_error}


def fix_agent_code(state, code_snippet_key, error_key, llm, prompt_template, function_name, retry_count_key="retry_count"):
    """
    Fix errors in the generated agent code using the LLM.
    
    Parameters
    ----------
    state : dict
        The current state containing code and error information.
    code_snippet_key : str
        Key in state where the broken code is stored.
    error_key : str
        Key in state where the error message is stored.
    llm : LLM
        The language model to use for fixing the code.
    prompt_template : str
        Template for the fix prompt (should have {code_snippet}, {error}, {function_name} placeholders).
    function_name : str
        Name of the function being fixed.
    retry_count_key : str, optional
        Key in state for tracking retry count. Defaults to "retry_count".
    
    Returns
    -------
    dict
        Dictionary with updated code, cleared error, and incremented retry count.
    """
    logger.info("Fixing agent code")
    logger.debug(f"Retry count: {state.get(retry_count_key)}")
    
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)
    
    # Create the fix prompt
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
    )
    
    # Get fixed code from LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }
