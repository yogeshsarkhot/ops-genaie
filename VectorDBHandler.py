import chromadb
import ollama
from typing import List, Dict

class VectorDBHandler:
    def __init__(self, collection_name: str = 'api_descriptions'):
        """
        Initialize ChromaDB vector database
        
        :param collection_name: Name of the collection to store embeddings
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using local Nomic Embed model
        
        :param texts: List of texts to embed
        :return: List of embeddings
        """
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model='nomic-embed-text', prompt=text)
            embeddings.append(response['embedding'])
        return embeddings
    
    def add_documents(self, apis: List[Dict[str, Any]]):
        """
        Add API descriptions to vector database
        
        :param apis: List of API descriptions
        """
        texts = [
            f"{api['unique_id']}: {api['summary']} {api['description']}" 
            for api in apis
        ]
        
        embeddings = self.generate_embeddings(texts)
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=[api['unique_id'] for api in apis]
        )
    
    def find_most_similar_api(self, query: str) -> str:
        """
        Find most similar API based on query
        
        :param query: User's query
        :return: Unique identifier of most similar API
        """
        query_embedding = self.generate_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        
        return results['ids'][0][0] if results['ids'] else None
