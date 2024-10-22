import requests
import json
from typing import List, Dict
from pathlib import Path

class FileInfo:
    def __init__(self, original: str, new_name: str, category: str, description: str, relative_path: str = ""):
        self.original = original
        self.new_name = new_name
        self.category = category
        self.description = description
        self.relative_path = relative_path

    def to_dict(self) -> Dict[str, str]:
        return {
            "original": self.original,
            "new_name": self.new_name,
            "category": self.category,
            "description": self.description,
            "relative_path": self.relative_path
        }

    def __str__(self):
        path_info = f" (in {self.relative_path})" if self.relative_path else ""
        return f"File: {self.original}{path_info} → {self.category}/{self.new_name}"

class FileOrganizerClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def preview_changes(self, 
                       folder_path: str, 
                       rename: bool = True, 
                       categorize: bool = True, 
                       cat_strength: int = 3, 
                       preferences: str = "",
                       keep_structure: bool = True) -> List[FileInfo]:
        url = f"{self.base_url}/preview"
        payload = {
            "folder_path": folder_path,
            "rename": rename,
            "categorize": categorize,
            "cat_strength": cat_strength,
            "preferences": preferences,
            "keep_structure": keep_structure
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return [FileInfo(**item) for item in response.json()]

    def apply_changes(self, folder_path: str, changes: List[FileInfo], keep_structure: bool = True):
        url = f"{self.base_url}/apply"
        payload = {
            "folder_path": folder_path,
            "changes": [change.to_dict() for change in changes],
            "keep_structure": keep_structure
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

def print_preview_summary(preview: List[FileInfo], keep_structure: bool):
    """Print a structured summary of the proposed changes."""
    print("\nProposed Changes Summary:")
    print("=" * 80)
    
    if keep_structure:

        by_path = {}
        for change in preview:
            path = change.relative_path or "root"
            if path not in by_path:
                by_path[path] = {}
            cat = change.category
            if cat not in by_path[path]:
                by_path[path][cat] = []
            by_path[path][cat].append(change)
        
        for path in sorted(by_path.keys()):
            print(f"\n📁 {path}")
            for category in sorted(by_path[path].keys()):
                print(f"  └─ 📂 {category}")
                for change in by_path[path][category]:
                    print(f"     └─ {change.original} → {change.new_name}")
    else:

        by_category = {}
        for change in preview:
            cat = change.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(change)
        
        for category in sorted(by_category.keys()):
            print(f"\n📂 {category}")
            for change in by_category[category]:
                orig_path = Path(change.relative_path) / change.original if change.relative_path else change.original
                print(f"  └─ {orig_path} → {change.new_name}")
    
    print("\nStatistics:")
    print(f"Total files to be processed: {len(preview)}")
    print(f"Number of categories: {len(set(change.category for change in preview))}")
    print("=" * 80)

def main():
    client = FileOrganizerClient()
    
    test_folder = "/Users/pranav/Documents/Projects/pristine-server/main_content"
    keep_structure = True  # Set this to False to flatten the directory structure
    
    print(f"Analyzing folder: {test_folder}")
    print(f"Keep folder structure: {'Yes' if keep_structure else 'No'}")
    
    try:
        preview = client.preview_changes(
            folder_path=test_folder,
            rename=True,
            categorize=True,
            cat_strength=1,
            preferences="",
            keep_structure=keep_structure
        )
        
        print_preview_summary(preview, keep_structure)
        
        while True:
            choice = input("\nDo you want to:\n"
                         "[a]pply these changes\n"
                         "[v]iew raw JSON preview\n"
                         "[c]ancel\n"
                         "Choice: ").lower()
            
            if choice == 'a':
                print("\nApplying changes...")
                result = client.apply_changes(test_folder, preview, keep_structure)
                print("Result:", result['message'])
                break
            elif choice == 'v':
                print("\nRaw preview data:")
                for change in preview:
                    print(json.dumps(change.to_dict(), indent=2))
            elif choice == 'c':
                print("\nOperation cancelled.")
                break
            else:
                print("\nInvalid choice, please try again.")
                
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        if hasattr(e.response, 'json'):
            print("Server response:", e.response.json())

if __name__ == "__main__":
    main()