import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "messages": [
        {"role": "user", "content": "3.8和3.11哪个更大"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
}

response = requests.post(url, json=data, headers=headers)
print(response.json()["choices"][0]["message"]["content"])