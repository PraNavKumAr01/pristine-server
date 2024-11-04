from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
import os
from PIL import Image
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from content_extractor import extract_content
from file_types import get_file_extension, image_exts
from typing import List
from chroma_new import CustomChroma
from clip_embeddings import ClipModelHandler
from text_embedding import SentenceTransformerHandler
from image_ocr import SmartOCR
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

text_db_path = str(Path.cwd() / "text_db")
image_db_path = str(Path.cwd() / "image_db")

text_handler = SentenceTransformerHandler()
model = text_handler.get_model()

clip_handler = ClipModelHandler()
clip_model = clip_handler.get_model()

ocr_handler = SmartOCR()

text_embeddings = HuggingFaceEmbeddings(client=model)

text_store = CustomChroma(
    embedding_function=text_embeddings, 
    collection_name='text_embeddings', 
    persist_directory=text_db_path
)
image_store = CustomChroma(
    embedding_function=clip_model,
    collection_name='image_embeddings',
    persist_directory=image_db_path
)

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

class VectorDBIds(BaseModel):
    text_ids: List[str]
    image_ids: List[str]
    total_ids: int

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
        
        if file_ext in image_exts:
            embedding = clip_model.encode(Image.open(file_path)).tolist()[0]
            image_store.add_embeddings(
                embeddings=[embedding],
                ids=[file_path]
            )
            return True
            
        content = extract_content(file_path)
        if len(content) > 1000:
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                id = f"{file_path}<chunk>{i}"
                text_store.add_texts(
                    texts=[chunk],
                    ids=[id]
                )
        else:
            text_store.add_texts(
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
    processed_files = {"text": 0, "image": 0}
    file_info = []
    skipped_files = []

    for path in paths:
        abs_path = os.path.abspath(path)
        files_to_process.extend(get_all_file_paths(abs_path))

    for file_path in tqdm(files_to_process, desc="Creating Embeddings and Adding to DB"):
        print(f"Processing {file_path}")
        
        file_ext = get_file_extension(file_path)
        if file_ext is None:
            print(f"Skipping unsupported file type: {file_path}")
            skipped_files.append(file_path)
            continue
            
        file_info_entry = {"path": file_path}
        
        try:
            if file_ext in image_exts:
                # Perform OCR
                ocr_text = ocr_handler.process_image(file_path)
                if ocr_text:
                    print(f"Doing OCR for {file_path}, found text {ocr_text}")
                    text_store.add_texts(
                            texts=[ocr_text],
                            metadatas=[{'path': file_path}],
                            ids=[file_path]
                        )
                embedding = clip_model.encode(Image.open(file_path)).tolist()[0]
                image_store.add_embeddings(
                    embeddings=[embedding],
                    metadatas=[{'path': file_path}],
                    ids=[file_path]
                )
                processed_files["image"] += 1
                file_info_entry["chunks"] = 1
            else:
                content = extract_content(file_path)
                if len(content) > 1000:
                    chunks = text_splitter.split_text(content)
                    for i, chunk in enumerate(chunks):
                        id = f"{file_path}<chunk>{i}"
                        text_store.add_texts(
                            texts=[chunk],
                            metadatas=[{'path': file_path}],
                            ids=[id]
                        )
                    file_info_entry["chunks"] = len(chunks)
                else:
                    text_store.add_texts(
                        texts=[content],
                        metadatas=[{'path': file_path}],
                        ids=[file_path]
                    )
                    file_info_entry["chunks"] = 1
                processed_files["text"] += 1
            file_info.append(file_info_entry)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            success = False

    return {
        "success": success,
        "total_files_processed": len(files_to_process) - len(skipped_files),
        "text_files_processed": processed_files["text"],
        "image_files_processed": processed_files["image"],
        "skipped_files": skipped_files,
        "file_info": file_info
    }

@app.post("/search", response_model=SearchResult)
async def search(query_model: QueryModel):
    query = query_model.query
    print(f"Searching for {query}")

    file_scores = {}
    
    results = text_store.similarity_search_with_score(query=query, k=10)
    
    for res in results:
        page, score = res
        if score <= 1.5:
            if '<chunk>' in page.metadata.get('path', ''):
                base_file_path = page.metadata['path'].split('<chunk>')[0]
            else:
                base_file_path = page.metadata['path']
            
            if base_file_path not in file_scores or score < file_scores[base_file_path]['score']:
                file_scores[base_file_path] = {
                    'score': score,
                    'type': 'text'
                }

    query_embedding = clip_model.encode(query).tolist()[0]
    print(query_embedding)
    image_results = image_store.similarity_search_with_score(
        query_embeddings=query_embedding,
        k=4
    )

    for image_res in image_results:
        page, score = image_res
        file_path = page.metadata['path']
        if score <= 185:
            if file_path not in file_scores or score < file_scores[file_path]['score']:
                file_scores[file_path] = {
                    'score': score,
                    'type': 'image'
                }

    print(f"File scores before sorting {file_scores}")

    sorted_results = sorted(
        file_scores.items(),
        key=lambda x: x[1]['score']
    )

    print(f"File scores after sorting {sorted_results}")

    sorted_file_ids = [file_path for file_path, _ in sorted_results]

    if sorted_file_ids:
        return SearchResult(file_paths=sorted_file_ids)
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
            file_ext = get_file_extension(file_path)
            if file_ext in image_exts:
                image_store.delete(ids=[file_path])
            else:
                ids_to_delete = [id for id in text_store.get()['ids'] if id.startswith(file_path)]
                text_store.delete(ids=ids_to_delete)
            deleted_paths.append(file_path)
        except Exception as e:
            print(f"Failed to delete embedding for {file_path}: {e}")
            failed_paths.append(file_path)

    return DeleteResult(deleted_paths=deleted_paths, failed_paths=failed_paths)

@app.post("/clear_vector_db", response_model=ClearResult)
def clear_vector_db():
    try:
        text_store.delete(ids=text_store.get()['ids'])
        image_store.delete(ids=image_store.get()['ids'])
        return ClearResult(success=True, message="Vector databases cleared successfully")
    except Exception as e:
        return ClearResult(success=False, message=f"Failed to clear vector databases: {str(e)}")
    
@app.get("/vector_db_ids", response_model=VectorDBIds)
def get_vector_db_ids():
    try:
        text_ids = text_store.get()['ids']
        image_ids = image_store.get()['ids']
        
        return VectorDBIds(
            text_ids=text_ids,
            image_ids=image_ids,
            total_ids=len(text_ids) + len(image_ids)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch vector database IDs: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
