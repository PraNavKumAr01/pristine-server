from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List
from file_types import get_file_extension
from content_extractor import extract_content
from ai_ops import suggest_filename, suggest_category

app = FastAPI()

class OrganizeRequest(BaseModel):
    folder_paths: List[str]
    rename: bool = True
    categorize: bool = True
    cat_strength: int = 1
    preferences: str = ""
    keep_structure: bool = True

class FileInfo(BaseModel):
    original: str
    new_name: str
    category: str
    description: str
    relative_path: str = ""
    parent_dir: str = ""

class ApplyRequest(BaseModel):
    folder_paths: List[str]
    changes: List[FileInfo]
    keep_structure: bool = True

class FileOrganizer:
    def __init__(self):
        self.file_mapping = {}
        self.all_categories = set()

    def get_all_files(self, folder_path: str, keep_structure: bool, base_path: str = None):
        """Recursively get all files with their relative paths."""
        if base_path is None:
            base_path = folder_path

        files_info = []
        
        for item in os.listdir(folder_path):
            if item == ".DS_Store":
                continue
                
            full_path = os.path.join(folder_path, item)
            
            if os.path.isfile(full_path):
                relative_path = os.path.relpath(os.path.dirname(full_path), base_path)
                relative_path = "" if relative_path == "." else relative_path
                files_info.append({
                    "filename": item,
                    "full_path": full_path,
                    "relative_path": relative_path,
                    "parent_dir": os.path.abspath(folder_path)
                })
            elif os.path.isdir(full_path):
                files_info.extend(self.get_all_files(full_path, keep_structure, base_path))
                
        return files_info

    def analyze_folders(self, folder_paths, rename=True, categorize=True, cat_strength=3, preferences="", keep_structure=True):
        preview_data = []
        for folder_path in folder_paths:
            self.file_mapping = {}
            self.all_categories = set()
            used_names = set()

            all_files = self.get_all_files(folder_path, keep_structure)

            if rename:
                for file_info in all_files:
                    file_ext = get_file_extension(file_info["full_path"])
                    content = file_info["full_path"] if file_ext in ['.png', '.jpg', '.jpeg'] else extract_content(file_info["full_path"])
                    new_content = suggest_filename(content, used_names, file_ext, preferences)
                    new_name = new_content["filename"] + file_ext
                    desc = new_content["description"]
                    used_names.add(os.path.splitext(new_name)[0])

                    original_relative = os.path.join(file_info["relative_path"], file_info["filename"])
                    self.file_mapping[original_relative] = {
                        "new_name": new_name,
                        "description": desc,
                        "category": "",
                        "relative_path": file_info["relative_path"],
                        "parent_dir": file_info["parent_dir"]
                    }

            if categorize:
                for file_info in all_files:
                    original_relative = os.path.join(file_info["relative_path"], file_info["filename"])
                    file_ext = get_file_extension(file_info["full_path"])
                    content = file_info["full_path"] if file_ext in ['.png', '.jpg', '.jpeg'] else extract_content(file_info["full_path"])
                    category = suggest_category(content, ",".join(self.all_categories), file_ext, cat_strength, preferences)
                    category = category.lower().replace(" ", "")
                    self.all_categories.add(category)
                    if original_relative in self.file_mapping:
                        self.file_mapping[original_relative]["category"] = category
                    else:
                        self.file_mapping[original_relative] = {
                            "new_name": file_info["filename"],
                            "description": "",
                            "category": category,
                            "relative_path": file_info["relative_path"],
                            "parent_dir": file_info["parent_dir"]
                        }

            preview_data.extend(self.generate_preview())
        return preview_data

    def generate_preview(self):
        preview_data = []
        for original, info in self.file_mapping.items():
            preview_data.append({
                "original": original,
                "new_name": info["new_name"],
                "category": info["category"],
                "description": info["description"],
                "relative_path": info["relative_path"],
                "parent_dir": info["parent_dir"]
            })
        return preview_data

    def apply_changes(self, changes: List[FileInfo], folder_paths: List[str], keep_structure: bool = True):
        print(changes)
        for folder_path in folder_paths:
            for change in changes:
                original_path = os.path.join(folder_path, change.original)
                
                if keep_structure:
                    category_path = os.path.join(folder_path, change.parent_dir, change.relative_path, change.category) if change.relative_path else os.path.join(folder_path, change.parent_dir, change.category)
                else:
                    category_path = os.path.join(folder_path, change.parent_dir, change.category)
                
                os.makedirs(category_path, exist_ok=True)
                
                new_path = os.path.join(category_path, change.new_name)
                
                counter = 1
                base_name, ext = os.path.splitext(change.new_name)
                while os.path.exists(new_path):
                    new_name = f"{base_name}_{counter}{ext}"
                    new_path = os.path.join(category_path, new_name)
                    counter += 1
                
                os.rename(original_path, new_path)

        return f"Applied changes to {len(changes)} files."

@app.post("/preview", response_model=List[FileInfo])
async def preview_changes(request: OrganizeRequest):
    for path in request.folder_paths:
        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail=f"Invalid folder path: {path}")
    
    organizer = FileOrganizer()
    preview_data = organizer.analyze_folders(
        request.folder_paths, 
        request.rename, 
        request.categorize, 
        request.cat_strength, 
        request.preferences,
        request.keep_structure
    )
    return preview_data

@app.post("/apply")
async def apply_changes(request: ApplyRequest):
    for path in request.folder_paths:
        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail=f"Invalid folder path: {path}")
    
    organizer = FileOrganizer()
    result = organizer.apply_changes(request.changes, request.folder_paths, request.keep_structure)
    return {"message": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
