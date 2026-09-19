import requests
import json

# 1. Define the Local API Endpoint
# By default, LM Studio runs an API endpoint on port 5000 or similar.
LOCAL_API_URL = "http://localhost:1234/api/v1/chat" 

def ask_llm(prompt):
    """Sends a structured request to the local LLM server and gets the response."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-lm-8ivs7xo6:O5diGa1QN5qw7lwl8NWn",
    }
    
    # This payload follows the standardized OpenAI API format.
    payload = {
        "model": "google/gemma-4-e4b", # Use the model name you loaded in LM Studio
        "input": prompt
    }

    try:
        # Make the POST request to the local server
        response = requests.post(LOCAL_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Throws an exception for bad status codes (4xx or 5xx)

        # The response is a JSON object; parse it into Python dictionary
        data = response.json()

        return data
        # # Extract the actual text content from the nested structure
        # if data and 'choices' in data and data['choices']:
        #     return data['choices'][0]['message']['content']
        # else:
        #     return "Error: Could not parse a valid response from the API."

    except requests.exceptions.ConnectionError:
        print("--- CONNECTION ERROR ---")
        print("Ensure LM Studio is running and the local server is active.")
        print(f"Attempted connection to {LOCAL_API_URL}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Usage Example ---
prompt = "What is the meaning of life?"
result = ask_llm(prompt)

if result:
    print("\n--- LLM RESPONSE ---\n")
    print(result)
