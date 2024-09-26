from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from content_extractor import extract_content
from summarizer import summarize_text, summarize_image
from file_types import get_file_extension, image_exts
from typing import List

app = FastAPI()

db_path = "/Users/pranav/Documents/Projects/smart-sort/image-src/db_new"
parent_path = "/Users/pranav/Documents/Projects/smart-sort/content/new_content"

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
store = Chroma(embedding_function=embeddings, collection_name='content_embeddings', persist_directory=db_path)

class FolderPath(BaseModel):
    path: str

class QueryModel(BaseModel):
    query: str
    parent_path: str

class SearchResult(BaseModel):
    file_paths: List[str]

@app.post("/add_to_vector_db")
def add_files_to_collection(folder: FolderPath):
    folder_path = folder.path
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)]
    success = True

    for file_path in tqdm(files, desc="Creating Embeddings and Adding to DB"):
        try:
            file_ext = get_file_extension(file_path)
            content = extract_content(file_path)
            if file_ext in image_exts:
                content = summarize_image(content)
            else:
                if len(content) >= 1000:
                    content = summarize_text(content)
            store.add_texts(
                texts=[content],
                ids=[file_path]
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            success = False

    return {"success": success}


@app.post("/search", response_model=SearchResult)
async def search(query_model: QueryModel):
    query = query_model.query
    print(f"Searching for {query}")

    contents = []
    file_ids = []
    encountered_files = set()

    results = store.similarity_search_with_score(query=query)
    for res in results:
        page, score = res
        content = page.page_content
        if score <= 1.5:
            contents.append(content)

    store_info = store.get()
    for content in contents:
        try:
            index = store_info['documents'].index(content)
            corresponding_id = store_info['ids'][index]

            normalized_id = corresponding_id.replace(query_model.parent_path, '').lstrip('/')

            if normalized_id not in encountered_files:
                encountered_files.add(normalized_id)
                full_path = os.path.join(query_model.parent_path, normalized_id)
                file_ids.append(full_path)

            print(f'The corresponding ID is: {full_path}')
        except ValueError:
            print('The text was not found in the documents list.')

    if file_ids:
        return SearchResult(file_paths=file_ids)
    else:
        raise HTTPException(status_code=404, detail="No relevant files found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
