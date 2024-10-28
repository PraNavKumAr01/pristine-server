import requests
import time
from typing import Optional, Dict, Any
import sys

class OllamaClient:
    def __init__(self, base_url: str = "http://0.0.0.0:8081"):
        """Initialize the Ollama client with the API base URL"""
        self.base_url = base_url.rstrip('/')
        
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an HTTP request to the API"""
        url = f"{self.base_url}/{endpoint}"
        try:
            print(f"Making {method} request to: {url}")
            if method == "GET":
                response = requests.get(url)
            else:
                print(f"Request data: {data}")
                response = requests.post(url, json=data)
            
            if not response.ok:
                print(f"Error response: {response.text}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Error details: {e.response.text}")
            sys.exit(1)

    def initialize(self, llm_name: str, install_path: Optional[str] = None) -> Dict[str, Any]:
        """Initialize the Ollama manager"""
        data = {"llm_name": llm_name}
        if install_path:
            data["install_path"] = install_path
        return self._make_request("POST", "init", data)

    def install(self) -> Dict[str, Any]:
        """Install Ollama"""
        return self._make_request("POST", "install")

    def start_server(self) -> Dict[str, Any]:
        """Start the Ollama server"""
        return self._make_request("POST", "start")

    def stop_server(self) -> Dict[str, Any]:
        """Stop the Ollama server"""
        return self._make_request("POST", "stop")

    def pull_model(self) -> Dict[str, Any]:
        """Pull the specified model"""
        return self._make_request("POST", "pull-model")

    def get_status(self) -> Dict[str, Any]:
        """Get the server status"""
        return self._make_request("GET", "status")

def test_ollama_functionality():
    """Test all Ollama management functionality"""
    client = OllamaClient()
    
    print("\n=== Testing Ollama Management API ===\n")

    try:
        print("1. Initializing Ollama manager...")
        result = client.initialize(llm_name="llama2")
        print(f"Result: {result}")

        print("\n2. Installing Ollama...")
        result = client.install()
        print(f"Result: {result}")

        print("\n3. Starting Ollama server...")
        result = client.start_server()
        print(f"Result: {result}")

        print("\n4. Checking server status...")
        result = client.get_status()
        print(f"Result: {result}")

        print("\n5. Pulling model...")
        #result = client.pull_model()
        #print(f"Result: {result}")

        # Wait a bit to let the model pull complete
        #time.sleep(5)

        print("\n6. Checking server status again...")
        result = client.get_status()
        print(f"Result: {result}")

        print("\n7. Stopping Ollama server...")
        result = client.stop_server()
        print(f"Result: {result}")

        print("\n8. Final status check...")
        result = client.get_status()
        print(f"Result: {result}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_ollama_functionality()