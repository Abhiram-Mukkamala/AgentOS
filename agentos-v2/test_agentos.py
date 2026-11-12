import requests
import json
import time

BASE_URL = "http://localhost:8080"

def run_test(name, prompt):
    print(f"\n{'='*50}\n[TEST] {name}\n{'='*50}")
    print(f"Prompt: {prompt}\n")
    try:
        response = requests.post(f"{BASE_URL}/chat", json={"prompt": prompt})
        if response.status_code == 200:
            print(f"Response:\n{response.json().get('response')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def main():
    # Wait for server to start
    print("Waiting for server to be ready...")
    for _ in range(10):
        try:
            if requests.get(f"{BASE_URL}/health").status_code == 200:
                print("Server is up!")
                break
        except:
            time.sleep(2)
    else:
        print("Server failed to start in time.")
        return

    # Test General Supervisor
    run_test("1. General Supervisor", "Hi! I'm an engineer. What can you help me with today?")

    # Test DBAgent
    run_test("2. DB Agent", "Please save this snippet for me. Title: 'Hello World Python'. Code: `print('Hello World!')`")

    # Test ResearchAgent
    run_test("3. Research Agent", "Can you search the web and tell me the time complexity of Timsort?")

    # Test Codex Agent - Level 1
    run_test("4. Codex Agent (Medium)", "Write a python function that finds the longest palindromic substring in a string. Execute it with the input 'babad' to verify it works.")

    # Test Codex Agent - Level 2
    run_test("5. Codex Agent (Hard)", "Write a multithreaded python script using `concurrent.futures` that calculates the factorial of numbers [500, 600, 700] in parallel. Execute it and print the length of the string representations of the results to verify it works.")

    # Test Codex Agent - Rate Original AgentOS
    original_agentos_description = '''
Original AgentOS was a simple FastAPI app that used a single chat endpoint. It had 3 tools (Python REPL, Wikipedia, and DB save) but it didn't have a real IDE. The frontend was just a chat box and didn't actually have file tabs or a Monaco editor, despite the instructions saying it did. Also, the database file initialized connections twice and failed to use the NullPool properly.
'''
    run_test("6. Codex Agent (Rate Original)", f"As the Codex Agent, analyze this description of the original AgentOS version and rate it on a scale of 1-10 for architectural soundness. Explain your rating: {original_agentos_description}")

if __name__ == "__main__":
    main()
