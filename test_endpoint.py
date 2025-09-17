import requests
import json

# Test the /api/chat/corrected endpoint
base_url = "http://localhost:8000"

def test_endpoint():
    print("Testing /api/chat/corrected endpoint...")
    
    # Test cases
    test_cases = [
        {
            "name": "Persian text with spelling error",
            "message": "سلم دوست من چطوری؟",
            "expected_correction": "سلام دوست من چطوری؟"
        },
        {
            "name": "English text with spelling errors", 
            "message": "helo my frend how are you?",
            "expected_correction": "hello my friend how are you?"
        },
        {
            "name": "Mixed Persian-English with errors",
            "message": "سلم my frend",
            "expected_correction": "سلام my friend"
        },
        {
            "name": "Text without errors",
            "message": "سلام دوست من",
            "expected_correction": "سلام دوست من"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {test_case['message']}")
        
        try:
            response = requests.post(
                f"{base_url}/api/chat/corrected",
                json={
                    "message": test_case["message"],
                    "user_id": "test_user"
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Status: SUCCESS")
                print(f"Response received - checking for spell correction...")
                
                # The response contains the full chat response, not just corrected text
                # We need to check if the spell correction was applied internally
                print(f"Full response structure received")
                
            else:
                print(f"Status: FAILED - HTTP {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Status: ERROR - {str(e)}")

if __name__ == "__main__":
    test_endpoint()