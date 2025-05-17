# Testing

Pull first the models:

```
$ ollama list
NAME                         ID              SIZE      MODIFIED       
llama3:8b-instruct-q4_0      365c0bd3c000    4.7 GB    37 seconds ago    
llama3.1:8b-instruct-q8_0    b158ded76fa0    8.5 GB    2 weeks ago       
llama3.2:3b                  a80c4f17acd5    2.0 GB    2 weeks ago       

$ python bench.py

🚀 Testing model: llama3.2:3b
⏱️ Warming up...
🔁 Run 1/5
🔁 Run 2/5
🔁 Run 3/5
🔁 Run 4/5
🔁 Run 5/5

📊 Results for model: llama3.2:3b
🧪 Outputs consistent across runs: ✅ YES
⏱️ Mean time: 4.99 sec
📈 Std deviation time: 0.05 sec
⚡ Mean tokens/sec: 71.94
📉 Std deviation tokens/sec: 0.75

🚀 Testing model: ollama.com/library/llama3:8b-instruct-q4_0
⏱️ Warming up...
🔁 Run 1/5
🔁 Run 2/5
🔁 Run 3/5
🔁 Run 4/5
🔁 Run 5/5

📊 Results for model: ollama.com/library/llama3:8b-instruct-q4_0
🧪 Outputs consistent across runs: ✅ YES
⏱️ Mean time: 13.90 sec
📈 Std deviation time: 0.01 sec
⚡ Mean tokens/sec: 34.25
📉 Std deviation tokens/sec: 0.02

🚀 Testing model: llama3.1:8b-instruct-q8_0
⏱️ Warming up...
🔁 Run 1/5
🔁 Run 2/5
🔁 Run 3/5
🔁 Run 4/5
🔁 Run 5/5

📊 Results for model: llama3.1:8b-instruct-q8_0
🧪 Outputs consistent across runs: ✅ YES
⏱️ Mean time: 39.65 sec
📈 Std deviation time: 2.77 sec
⚡ Mean tokens/sec: 9.32
📉 Std deviation tokens/sec: 0.62

Some models may not fit in memory depending on your card which affects performance:

ollama ps
NAME                         ID              SIZE      PROCESSOR          UNTIL              
llama3.1:8b-instruct-q8_0    b158ded76fa0    9.7 GB    21%/79% CPU/GPU    4 minutes from now

Tests were run on a RTX-4060, 8GB
```