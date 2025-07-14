import requests
import json

url = "http://localhost:8000/api/chat"

payload = json.dumps({
  "message": "روش های کود دهی",
  "user_id": "test_user"
})

headers = {
  'Content-Type': 'application/json'
}

try:
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
except requests.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")