import requests
import time
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"  # or llama2, or your custom model tag

PROMPT = """Explain the concept of vector embeddings in natural language processing. 
Use simple language and provide a real-world example."""

def get_response(prompt, model="mistral"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    start_time = time.time()
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
    end_time = time.time()

    if response.status_code == 200:
        result = response.json()
        output_text = result.get("response", "")
        duration = end_time - start_time
        word_count = len(output_text.strip().split())
        token_count = word_count  # Rough estimate: 1 word ≈ 1 token
        tokens_per_sec = token_count / duration

        print(f"\n🧠 Prompt:\n{prompt}\n")
        print(f"📤 Response:\n{output_text}\n")
        print(f"⏱️ Time taken: {duration:.2f} seconds")
        print(f"🔢 Tokens (approx): {token_count}")
        print(f"⚡ Tokens/sec: {tokens_per_sec:.2f}")
    else:
        print("❌ Request failed:", response.status_code, response.text)

if __name__ == "__main__":
    get_response(PROMPT, MODEL_NAME)