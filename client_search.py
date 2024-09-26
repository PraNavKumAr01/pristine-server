import requests
import json

BASE_URL = "http://localhost:8000"

def add_to_vector_db(folder_path):
    url = f"{BASE_URL}/add_to_vector_db"
    payload = {"path": folder_path}
    response = requests.post(url, json=payload)
    return response.json()

def search_files(query, parent_path):
    url = f"{BASE_URL}/search"
    payload = {"query": query, "parent_path": parent_path}
    response = requests.post(url, json=payload)
    return response.json()

def main():

    folder_path = "/Users/pranav/Documents/Projects/smart-sort/content/new_content"
    print("Adding files to vector DB...")
    result = add_to_vector_db(folder_path)
    print(f"Result: {result}")

    query = "car"
    parent_path = "/Users/pranav/Documents/Projects/smart-sort/content/new_content"
    print("\nSearching files...")
    try:
        result = search_files(query, parent_path)
        print("Search results:")
        for file_path in result['file_paths']:
            print(f"- {file_path}")
    except requests.exceptions.HTTPError as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    main()