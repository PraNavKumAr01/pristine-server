import uuid
from langchain_community.vectorstores import Chroma
from typing import List, Optional, Any, Tuple, Dict
from langchain_core.documents import Document
from langchain_core.utils import xor_args

def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

class CustomChroma(Chroma):

    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Query the chroma collection."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add pre-computed embeddings to the vectorstore.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors
            texts (Optional[List[str]]): Optional list of texts corresponding to embeddings
            metadatas (Optional[List[dict]]): Optional list of metadatas
            ids (Optional[List[str]]): Optional list of IDs
            
        Returns:
            List[str]: List of IDs of the added embeddings
        """
        if texts is None:
            texts = [""] * len(embeddings)  # Empty texts for image embeddings
            
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]
            
        if metadatas:
            # fill metadatas with empty dicts if needed
            length_diff = len(embeddings) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
                
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
                    
            if non_empty_ids:
                filtered_metadatas = [metadatas[idx] for idx in non_empty_ids]
                filtered_texts = [texts[idx] for idx in non_empty_ids]
                filtered_embeddings = [embeddings[idx] for idx in non_empty_ids]
                filtered_ids = [ids[idx] for idx in non_empty_ids]
                
                try:
                    self._collection.upsert(
                        metadatas=filtered_metadatas,
                        embeddings=filtered_embeddings,
                        documents=filtered_texts,
                        ids=filtered_ids,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
                        
            if empty_ids:
                remaining_texts = [texts[j] for j in empty_ids]
                remaining_embeddings = [embeddings[j] for j in empty_ids]
                remaining_ids = [ids[j] for j in empty_ids]
                
                self._collection.upsert(
                    embeddings=remaining_embeddings,
                    documents=remaining_texts,
                    ids=remaining_ids,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids
    
    def similarity_search_with_score(
        self,
        query: Optional[str] = None,
        query_embeddings: Optional[List[float]] = None,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (Optional[str]): Query text to search for.
            query_embeddings (Optional[List[float]]): Pre-computed query embeddings.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query and cosine distance in float for each.
            Lower score represents more similarity.
        """
        if query_embeddings is not None:
            results = self.__query_collection(
                query_embeddings=[query_embeddings],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )
        elif self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )

        return _results_to_docs_and_scores(results)
