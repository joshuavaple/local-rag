# client.py
import requests
import argparse



# Base URL of your local FastAPI service
BASE_URL = "http://127.0.0.1:8000" # this is localhost
# BASE_URL = "http://localhost:8000" # this is also ok

# Endpoint to embed texts
ENDPOINT = f"{BASE_URL}/embed"

# Example texts to embed
sample_texts = [
        "Hello world",
        "Embedding is a vector"
    ]

def invoke_embedding_endpoint(texts: list[str]):
    payload = {
        "texts": texts
    }
    headers = {
        "Content-Type": "application/json"
    }
    # Make the POST request
    response = requests.post(ENDPOINT, json=payload, headers=headers)
    # Check response status
    if response.status_code == 200:
        data = response.json()
        print(data)

    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

def main():
    parser = argparse.ArgumentParser(description="Call local embedding service")
    parser.add_argument(
        "texts",
        nargs="+",
        help="One or more text strings to embed (separate with space, wrap multi-word text in quotes)",
    )
    args = parser.parse_args()

    invoke_embedding_endpoint(args.texts)

if __name__ == "__main__":
    # invoke_embedding_endpoint(texts=sample_texts)
    main()