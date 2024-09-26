import requests
import json
from typing import List, Dict

class FileInfo:
    def __init__(self, original: str, new_name: str, category: str, description: str):
        self.original = original
        self.new_name = new_name
        self.category = category
        self.description = description

    def to_dict(self) -> Dict[str, str]:
        return {
            "original": self.original,
            "new_name": self.new_name,
            "category": self.category,
            "description": self.description
        }

class FileOrganizerClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def preview_changes(self, folder_path, rename=True, categorize=True, cat_strength=3, preferences=""):
        url = f"{self.base_url}/preview"
        payload = {
            "folder_path": folder_path,
            "rename": rename,
            "categorize": categorize,
            "cat_strength": cat_strength,
            "preferences": preferences
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return [FileInfo(**item) for item in response.json()]

    def apply_changes(self, folder_path: str, changes: List[FileInfo]):
        url = f"{self.base_url}/apply"
        payload = {
            "folder_path": folder_path,
            "changes": [change.to_dict() for change in changes]
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

def main():
    client = FileOrganizerClient()

    test_folder = "/Users/pranav/Documents/Projects/local-ollama/content/main_content"

    print("Previewing changes...")
    preview = client.preview_changes(
        folder_path=test_folder,
        rename=True,
        categorize=True,
        cat_strength=1,
        preferences=""
    )

    print("Preview of changes:")
    for change in preview:
        print(json.dumps(change.to_dict(), indent=2))

    apply_changes = input("Do you want to apply these changes? (yes/no): ").lower() == 'yes'

    if apply_changes:
        print("Applying changes...")
        result = client.apply_changes(test_folder, preview)
        print("Result:", result['message'])
    else:
        print("Changes not applied.")

if __name__ == "__main__":
    main()