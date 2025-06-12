import json
import statistics
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAMES = [
    "llama3.2:3b",
    "ollama.com/library/llama3:8b-instruct-q4_0",
    "llama3.1:8b-instruct-q8_0",
]
RUNS_PER_MODEL = 5
PROMPT = """Explain the concept of vector embeddings in natural language processing. 
Use simple language and provide a real-world example."""


def get_response(prompt, model="mistral", warm_up=False):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "n_predict": 128,
            "n_probs": 1,
            "temperature": 0,
            "samplers": ["temperature"],
            "seed": 1234,
            "repeat_last_n": 0,
            "min_p": 0.0,
            "top_p": 1.0,
            "top_k": 100,
            "repeat_penalty": 1.0,
            "mirostat_eta": 0.0,
            "mirostat_tau": 0.0,
            "cache_prompt": False,
        },
    }

    start_time = time.time()
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
    end_time = time.time()
    if warm_up:
        return

    if response.status_code == 200:
        result = response.json()
        output_text = result.get("response", "")
        duration = end_time - start_time
        word_count = len(output_text.strip().split())
        token_count = word_count  # Rough estimate: 1 word ≈ 1 token
        tokens_per_sec = token_count / duration

        # print(f"\n🧠 Prompt:\n{prompt}\n")
        # print(f"📤 Response:\n{output_text}\n")
        # print(f"⏱️ Time taken: {duration:.2f} seconds")
        # print(f"🔢 Tokens (approx): {token_count}")
        # print(f"⚡ Tokens/sec: {tokens_per_sec:.2f}")
        return output_text, duration, tokens_per_sec
    else:
        print("❌ Request failed:", response.status_code, response.text)
        return None, None, None


if __name__ == "__main__":
    # for model in MODEL_NAME:
    #     # warm up
    #     get_response(PROMPT, model, True)

    #     get_response(PROMPT, model, False)

    for model in MODEL_NAMES:
        print(f"\n🚀 Testing model: {model}")
        outputs = []
        durations = []
        tps_list = []

        # Warm-up
        print("⏱️ Warming up...")
        get_response(PROMPT, model)

        for i in range(RUNS_PER_MODEL):
            print(f"🔁 Run {i + 1}/{RUNS_PER_MODEL}")
            output, duration, tps = get_response(PROMPT, model)
            if output is None:
                continue
            outputs.append(output)
            durations.append(duration)
            tps_list.append(tps)

        # Check for consistency
        unique_outputs = set(outputs)
        outputs_consistent = len(unique_outputs) == 1

        # Report
        print(f"\n📊 Results for model: {model}")
        print(
            f"🧪 Outputs consistent across runs: {'✅ YES' if outputs_consistent else '❌ NO'}"
        )
        print(f"⏱️ Mean time: {statistics.mean(durations):.2f} sec")
        print(
            f"📈 Std deviation time: {statistics.stdev(durations):.2f} sec"
            if len(durations) > 1
            else "📈 Not enough data for std deviation"
        )
        print(f"⚡ Mean tokens/sec: {statistics.mean(tps_list):.2f}")
        print(
            f"📉 Std deviation tokens/sec: {statistics.stdev(tps_list):.2f}"
            if len(tps_list) > 1
            else ""
        )
