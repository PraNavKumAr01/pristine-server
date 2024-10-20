from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List
from file_types import get_file_extension
from content_extractor import extract_content
from ai_ops import suggest_filename, suggest_category

app = FastAPI()

class OrganizeRequest(BaseModel):
    folder_path: str
    rename: bool = False
    categorize: bool = False
    cat_strength: int = 5
    preferences: str = ""

class FileInfo(BaseModel):
    original: str
    new_name: str
    category: str
    description: str

class ApplyRequest(BaseModel):
    folder_path: str
    changes: List[FileInfo]

class FileOrganizer:
    def __init__(self):
        self.file_mapping = {}
        self.all_categories = set()

    def analyze_folder(self, folder_path, rename=True, categorize=True, cat_strength=3, preferences=""):
        self.file_mapping = {}
        self.all_categories = set()
        used_names = set()

        if rename:
            for filename in os.listdir(folder_path):
                if filename == ".DS_Store":
                    continue
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    content = extract_content(file_path)
                    file_ext = get_file_extension(file_path)
                    new_content = suggest_filename(content, used_names, file_ext, preferences)
                    print(new_content)
                    new_name = new_content["filename"]
                    new_name += file_ext
                    desc = new_content["description"]
                    used_names.add(os.path.splitext(new_name)[0])
                    self.file_mapping[filename] = {
                        "new_name": new_name,
                        "description": desc, 
                        "category": "",
                    }

        if categorize:
            for filename in os.listdir(folder_path):
                if filename == ".DS_Store":
                    continue
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    content = extract_content(file_path)
                    category = suggest_category(content, ",".join(self.all_categories), get_file_extension(file_path), cat_strength, preferences)
                    category = category.lower().replace(" ", "")
                    self.all_categories.add(category)
                    
                    if filename in self.file_mapping:
                        self.file_mapping[filename]["category"] = category
                    else:
                        self.file_mapping[filename] = {
                            "new_name": filename,
                            "description": "", 
                            "category": category,
                        }

        return self.generate_preview()

    def generate_preview(self):
        preview_data = []
        for original, info in self.file_mapping.items():
            preview_data.append({
                "original": original,
                "new_name": info["new_name"],
                "category": info["category"],
                "description": info["description"],
            })
        return preview_data

    def apply_changes(self, changes: List[FileInfo], folder_path: str):
        for change in changes:
            original_path = os.path.join(folder_path, change.original)
            category_path = os.path.join(folder_path, change.category)
            os.makedirs(category_path, exist_ok=True)
            new_path = os.path.join(category_path, change.new_name)

            os.rename(original_path, os.path.join(folder_path, change.new_name))
            os.rename(os.path.join(folder_path, change.new_name), new_path)

        return f"Applied changes to {len(changes)} files."

@app.post("/preview", response_model=List[FileInfo])
async def preview_changes(request: OrganizeRequest):
    if not os.path.isdir(request.folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path")
    
    organizer = FileOrganizer()
    preview_data = organizer.analyze_folder(
        request.folder_path, 
        request.rename, 
        request.categorize, 
        request.cat_strength, 
        request.preferences
    )
    return preview_data

@app.post("/apply")
async def apply_changes(request: ApplyRequest):
    if not os.path.isdir(request.folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path")
    
    organizer = FileOrganizer()
    result = organizer.apply_changes(request.changes, request.folder_path)
    return {"message": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)