from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from content_extractor import extract_content
from file_types import get_file_extension, image_exts
from typing import List, Optional, Any

app = FastAPI()

db_path = "/Users/pranav/Documents/Projects/smart-sort/image-src/db_new"

model = SentenceTransformer(
   "/Users/pranav/Documents/Projects/pristine-server/embedding_model",
   backend="onnx",
   model_kwargs={
      "file_name": "model_qint8_avx512_vnni.onnx",
      "provider" : "CPUExecutionProvider"
  },
)

embeddings = HuggingFaceEmbeddings(client = model)
store = Chroma(embedding_function=embeddings, collection_name='content_embeddings', persist_directory=db_path)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

class PathsModel(BaseModel):
    paths: List[str]

class QueryModel(BaseModel):
    query: str

class SearchResult(BaseModel):
    file_paths: List[str]

class DeleteResult(BaseModel):
    deleted_paths: List[str]
    failed_paths: List[str]

class ClearResult(BaseModel):
    success: bool
    message: str

def is_system_file(filename: str) -> bool:
    """Check if a file is a system file or hidden file."""
    system_files = {'.DS_Store', 'Thumbs.db', '.gitignore', '.git'}
    return filename.startswith('.') or filename in system_files

def get_all_file_paths(path: str) -> List[str]:
    if os.path.isfile(path):
        return [os.path.abspath(path)] if not is_system_file(os.path.basename(path)) else []
    elif os.path.isdir(path):
        file_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                if not is_system_file(file):
                    file_paths.append(os.path.abspath(os.path.join(root, file)))
        return file_paths
    else:
        return []

def process_file(file_path: str) -> bool:
    try:
        file_ext = get_file_extension(file_path)
        content = extract_content(file_path)
        
        if len(content) > 1000:
            chunks = text_splitter.split_text(content)
            print(len(chunks))
            for i, chunk in enumerate(chunks):
                id = f"{file_path}<chunk>{i}"
                print(id)
                store.add_texts(
                    texts=[chunk],
                    ids=[id]
                )
        else:
            store.add_texts(
                texts=[content],
                ids=[file_path]
            )
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

@app.post("/add_to_vector_db")
def add_to_collection(paths_model: PathsModel):
    paths = paths_model.paths
    success = True
    files_to_process = []

    for path in paths:
        abs_path = os.path.abspath(path)
        files_to_process.extend(get_all_file_paths(abs_path))

    for file_path in tqdm(files_to_process, desc="Creating Embeddings and Adding to DB"):
        print(f"Embeddings {file_path}")
        if not process_file(file_path):
            success = False

    return {"success": success, "files_processed": len(files_to_process)}

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

            # Extract the base file path (remove the chunk identifier if present)
            base_file_path = corresponding_id.split('<chunk>')[0]

            if base_file_path not in encountered_files:
                encountered_files.add(base_file_path)
                file_ids.append(base_file_path)

            print(f'The corresponding ID is: {corresponding_id}')
        except ValueError:
            print('The text was not found in the documents list.')

    if file_ids:
        return SearchResult(file_paths=file_ids)
    else:
        raise HTTPException(status_code=404, detail="No relevant files found.")

@app.post("/delete_embeddings", response_model=DeleteResult)
def delete_embeddings(paths_model: PathsModel):
    paths = paths_model.paths
    deleted_paths = []
    failed_paths = []

    all_file_paths = []
    for path in paths:
        all_file_paths.extend(get_all_file_paths(path))

    for file_path in tqdm(all_file_paths, desc="Deleting embeddings"):
        try:
            # Delete all chunks associated with this file
            ids_to_delete = [id for id in store.get()['ids'] if id.startswith(file_path)]
            store.delete(ids=ids_to_delete)
            deleted_paths.append(file_path)
        except Exception as e:
            print(f"Failed to delete embedding for {file_path}: {e}")
            failed_paths.append(file_path)

    return DeleteResult(deleted_paths=deleted_paths, failed_paths=failed_paths)

@app.post("/clear_vector_db", response_model=ClearResult)
def clear_vector_db():
    try:
        # Get all IDs in the database
        all_ids = store.get()['ids']
        
        # Delete all documents
        store.delete(ids=all_ids)
        
        return ClearResult(success=True, message="Vector database cleared successfully")
    except Exception as e:
        return ClearResult(success=False, message=f"Failed to clear vector database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)