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

def clear_vector_db():
    url = f"{BASE_URL}/clear_vector_db"
    response = requests.post(url)
    return response.json()

def get_db_ids():
    url = f"{BASE_URL}/vector_db_ids"
    response = requests.get(url)
    return response.json()

def main():
    # Test add_to_vector_db
    folder_paths = ["/Users/pranav/Documents/Projects/pristine-server/main_content"]
    print("Adding files to vector DB...")
    result = add_to_vector_db(folder_paths)
    print(f"Add result: {result}")

    get_res = get_db_ids()
    #print(get_res)

    # Test search_files
    query = "Physical Activity"
    print("\nSearching files...")
    try:
        result = search_files(query)
        print("Search results:")
        try:
            for file_path in result['file_paths']:
                print(f"- {file_path}")
        except ValueError as e:
            print("No results found")
    except requests.exceptions.HTTPError as e:
        print(f"Error during search: {e}")

    # Test delete_embeddings
    paths_to_delete = ["/Users/pranav/Documents/Projects/pristine-server/main_content/climate-change/climate_change.txt"]
    print("\nDeleting embeddings...")
    delete_result = delete_embeddings(paths_to_delete)
    print(f"Delete result: {delete_result}")

    # Test search_files again to verify deletion
    print("\nSearching files after deletion...")
    try:
        result = search_files(query)
        print("Search results after deletion:")
        for file_path in result['file_paths']:
            print(f"- {file_path}")
    except requests.exceptions.HTTPError as e:
        print(f"Error during search: {e}")

    # Test clear_vector_db
    print("\nClearing vector DB...")
    clear_result = clear_vector_db()
    print(f"Clear result: {clear_result}")

    # Test search_files one last time to verify clearing
    print("\nSearching files after clearing DB...")
    try:
        result = search_files(query)
        print("Search results after clearing DB:")
        try:
            if result['file_paths']:
                for file_path in result['file_paths']:
                    print(f"- {file_path}")
            else:
                print("No results Found")
        except KeyError as e:
            print("No results Found")
    except requests.exceptions.HTTPError as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    main()
