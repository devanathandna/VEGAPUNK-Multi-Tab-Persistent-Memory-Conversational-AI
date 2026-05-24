import requests
import json

print("Testing VEGAPUNK Backend API...")
print("=" * 60)

# Test 1: Health check
print("\n[TEST 1] Health Check")
try:
    response = requests.get("http://localhost:5000/api/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Send a chat message
print("\n[TEST 2] Send Chat Message")
try:
    response = requests.post(
        "http://localhost:5000/api/chat/main",
        json={"message": "Hello, test message"},
        headers={"Content-Type": "application/json"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Get sessions
print("\n[TEST 3] Get Sessions")
try:
    response = requests.get("http://localhost:5000/api/sessions/grouped")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
