# Marketing Email Generator

## 0. Setup and config

Start new virtual environment:

    python -m venv /path/to/new/venv

Activate virtual environment

    source /path/to/venv/bin/activate

Navigate to repo base folder and run:

    pip install r requirements.txt


## Generate API key

Generate Open AI API key
    https://platform.openai.com/account/api-keys

Make a copy of .env_template file name it .env

paste the API key into the newly created .env

## 2. Run Streamlit app

From repo base folder run fom command line:

    streamlit run app.py