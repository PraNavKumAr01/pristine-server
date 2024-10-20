import requests
import json
from typing import List

BASE_URL = "http://localhost:8000"

def add_to_vector_db(paths: List[str]):
    url = f"{BASE_URL}/add_to_vector_db"
    payload = {"paths": paths}
    response = requests.post(url, json=payload)
    return response.json()

def search_files(query: str):
    url = f"{BASE_URL}/search"
    payload = {"query": query}
    response = requests.post(url, json=payload)
    return response.json()

def delete_embeddings(paths: List[str]):
    url = f"{BASE_URL}/delete_embeddings"
    payload = {"paths": paths}
    response = requests.post(url, json=payload)
    return response.json()

def main():

    folder_paths = ["/Users/pranav/Documents/Projects/pristine-server/main_content"]
    print("Adding files to vector DB...")
    result = add_to_vector_db(folder_paths)
    print(f"Result: {result}")

    query = "climate change"
    print("\nSearching files...")
    try:
        result = search_files(query)
        print("Search results:")
        for file_path in result['file_paths']:
            print(f"- {file_path}")
    except requests.exceptions.HTTPError as e:
        print(f"Error during search: {e}")

    paths_to_delete = ["/Users/pranav/Documents/Projects/pristine-server/main_content/climate-change/climate_change.txt"]
    print("\nDeleting embeddings...")
    delete_result = delete_embeddings(paths_to_delete)
    print(f"Delete result: {delete_result}")

if __name__ == "__main__":
    main()