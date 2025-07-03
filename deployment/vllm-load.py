import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

PAYLOAD = {
    "model": "/model",
    "messages": [
        {"role": "user", "content": "What do you know about the capital of Greece? Summarize in a paragraph."}
    ],
    "max_tokens": 100,
    "temperature": 0.0
}

def send_request(session, i):
    try:
        start = time.time()
        response = session.post(URL, headers=HEADERS, data=json.dumps(PAYLOAD))
        latency = time.time() - start
        return {
            "index": i,
            "status_code": response.status_code,
            "latency": latency,
            "error": None if response.ok else response.text[:200]
        }
    except Exception as e:
        return {"index": i, "status_code": None, "latency": None, "error": str(e)}

def run_load_test(concurrent_users=10, total_requests=50):
    results = []
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        with requests.Session() as session:
            futures = [
                executor.submit(send_request, session, i)
                for i in range(total_requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())

    return results

if __name__ == "__main__":
    concurrent_users = 10
    total_requests = 50

    print(f"Running load test with {concurrent_users} users and {total_requests} total requests...")
    results = run_load_test(concurrent_users, total_requests)

    success = [r for r in results if r["status_code"] == 200]
    failed = [r for r in results if r["status_code"] != 200]

    print(f"\n✅ Success: {len(success)}")
    print(f"❌ Failed: {len(failed)}")

    if success:
        avg_latency = sum(r["latency"] for r in success) / len(success)
        print(f"📊 Avg Latency: {avg_latency:.2f} seconds")

    if failed:
        print("\nErrors (first 5):")
        for r in failed[:5]:
            print(f"- {r['error']}")
