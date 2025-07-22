import json
import time

import requests

LLAMA_CPP_URL = "http://localhost:8080/v1/completions"
MODEL_NAME = "llama.cpp"  # for logging purposes only

PROMPT = """Explain the concept of vector embeddings in natural language processing. 
Use simple language and provide a real-world example."""


def get_response(prompt: str) -> None:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "any",  # llama.cpp ignores this value
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False,
    }

    start_time = time.time()
    response = requests.post(LLAMA_CPP_URL, headers=headers, data=json.dumps(payload))
    end_time = time.time()

    if response.status_code == 200:
        result = response.json()
        output_text = result.get("choices", [{}])[0].get("text", "")
        duration = end_time - start_time
        word_count = len(output_text.strip().split())
        token_count = word_count  # Approximate
        tokens_per_sec = token_count / duration

        print(f"\nPrompt:\n{prompt}\n")
        print(f"Response:\n{output_text.strip()}\n")
        print(f"Time taken: {duration:.2f} seconds")
        print(f"Tokens (approx): {token_count}")
        print(f"Tokens/sec: {tokens_per_sec:.2f}")
    else:
        print("Request failed:", response.status_code, response.text)


if __name__ == "__main__":
    get_response(PROMPT)
