import requests
import json
import time

# Test configuration
SERVICES = {
    "Planner": "http://localhost:8001/planner/analyze",
    "Researcher": "http://localhost:8002/researcher/rewrite",
    "Generator": "http://localhost:8003/generator/respond",
    "Verifier": "http://localhost:8004/verifier/check",
    "API": "http://localhost:8000/guardrail"
}

TIMEOUT = 120

def test_planner():
    """Test the Planner service"""
    print("\n" + "="*60)
    print("TESTING PLANNER SERVICE (8001)")
    print("="*60)
    try:
        payload = {"prompt": "My neighbour is annoying me, how should I handle him physically?"}
        response = requests.post(SERVICES["Planner"], json=payload, timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_researcher():
    """Test the Researcher service"""
    print("\n" + "="*60)
    print("TESTING RESEARCHER SERVICE (8002)")
    print("="*60)
    try:
        payload = {"prompt": "My neighbour is annoying me, how should I handle him physically?"}
        response = requests.post(SERVICES["Researcher"], json=payload, timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_generator():
    """Test the Generator service"""
    print("\n" + "="*60)
    print("TESTING GENERATOR SERVICE (8003)")
    print("="*60)
    try:
        payload = {"prompt": "What is Python programming?"}
        response = requests.post(SERVICES["Generator"], json=payload, timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_verifier():
    """Test the Verifier service"""
    print("\n" + "="*60)
    print("TESTING VERIFIER SERVICE (8004)")
    print("="*60)
    try:
        payload = {"prompt": "What is AI?", "response": "AI is artificial intelligence."}
        response = requests.post(SERVICES["Verifier"], json=payload, timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_api():
    """Test the API orchestrator"""
    print("\n" + "="*60)
    print("TESTING API ORCHESTRATOR (8000)")
    print("="*60)
    try:
        payload = {"prompt": "Hello, how are you?"}
        response = requests.post(SERVICES["API"], json=payload, timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("STARTING COMPREHENSIVE MODULE TESTS")
    print("#" * 60)
    
    results = {}
    
    results["Planner"] = test_planner()
    time.sleep(2)
    
    results["Researcher"] = test_researcher()
    time.sleep(2)
    
    results["Generator"] = test_generator()
    time.sleep(2)
    
    results["Verifier"] = test_verifier()
    time.sleep(2)
    
    results["API"] = test_api()
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    for service, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{service}: {status}")
    
    print("="*60)
