rm -rf .venv  # Clean existing virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt