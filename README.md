Backend logic to interact with the model (random currently to test connection) hosted on Hugging Face Spaces and the Frontend.

(!!MPORTANT!!) Ensure that your python version is 3.12.4
(!!MPORTANT!!) Ensure that your pip version is >= 24.0 (or latest)

1. cd into the project folder
2. run python -m venv .venv
3. run .venv\Scripts\Activate
4. run pip install -r requirements.txt
5. run uvicorn main:app --reload

Ensure that .venv is added to .gitignore to avoid pushing your local virtual environment
