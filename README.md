# Data Cleaning Agent

An AI-powered data cleaning agent that automatically cleans messy datasets using LangChain and LangGraph. The agent uses an LLM to generate and execute Python code for data cleaning tasks like handling missing values, removing duplicates, treating outliers, and dropping low-quality columns.

## How It Works

The agent follows a simple workflow:
1. **Inspect**: Scans your dataset for missing values and IQR outliers, column by column
2. **Decide**: You choose a cleaning strategy per column (or leave the default)
3. **Generate**: Uses an LLM to create custom Python cleaning code based on your decisions
4. **Execute**: Runs the generated code to clean your data
5. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

This approach combines the flexibility of LLMs with per-column control by the user.

## Setup

### Prerequisites

- **Python 3.9 or higher** (3.9, 3.10, 3.11, 3.12, or 3.13) - **Note**: Python 3.9.7 is not supported due to a Streamlit compatibility issue
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.9, python3.10, python3.12, etc.

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Streamlit Web Interface

The easiest way to use the agent is through the web interface:

```bash
poetry run streamlit run app.py
```

Then:
1. Upload your CSV file
2. Review the detected data quality issues (missing values and IQR outliers per column)
3. Choose a cleaning strategy for each affected column:
   - **Numeric columns — missing values**: basic cleaning / impute with mean / impute with median / drop rows / custom
   - **Numeric columns — outliers**: basic cleaning / replace with mean / replace with median / drop rows / custom
   - **Categorical columns — missing values**: basic cleaning / impute with mode / drop rows / custom
4. Click "Clean Data"
5. Download the cleaned dataset

Selecting **"basic cleaning"** (the default) applies the agent's built-in rules: drop columns with >40% missing values, impute remaining missing values (mean for numeric, mode for categorical), and remove duplicates. Selecting **"custom"** lets you type any free-text instruction which the LLM will interpret.

### Docker

To run the app without installing Python or Poetry:

```bash
# Build the image (once)
docker build -t data-cleaning-agent .

# Run the container
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-your-key-here data-cleaning-agent
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Python API

For programmatic use or integration into data pipelines:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize the agent with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)

# Load your messy data
df = pd.read_csv("your_data.csv")

# Run the cleaning agent
agent.invoke_agent(data_raw=df)

# Get the cleaned dataset
cleaned_df = agent.get_data_cleaned()

# Save or use the cleaned data
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

**Optional: provide custom instructions**

```python
agent.invoke_agent(
    data_raw=df,
    user_instructions="Remove columns with more than 30% missing values and standardize date formats"
)
```

**Optional: save the workflow graph as a PNG**

```python
agent.save_graph_visualization()          # saves to logs/workflow_graph.png
agent.save_graph_visualization("my/path/graph.png")  # custom path
```

> Requires `playwright` (`pip install playwright && playwright install`).

## Project Structure

```
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # Agent class + LangGraph workflow
│   └── utils.py                # Data quality analysis + utility functions
├── app.py                      # Streamlit interface
├── Dockerfile                  # Container image for cross-platform sharing
├── pyproject.toml              # Dependencies configuration
├── poetry.lock                 # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.
