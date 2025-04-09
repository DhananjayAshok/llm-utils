echo "Make sure you are running this script from project root"
pip install --upgrade uv
uv venv llm-utils-env
source llm-utils-env/bin/activate
pip install -r requirements.txt