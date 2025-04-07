import ollama
from app.database import Database
from app.embeddings import EmbeddingHandler

class LLMHandler:
    def __init__(self):
        self.model = "llama3"
        self.history = []
        self.db = Database()
        self.embedding_handler = EmbeddingHandler()

    def get_response(self, query):
        # First, try to find relevant information from vector database
        api_match = self.embedding_handler.search_similar(query)
        
        # Prepare the system prompt to restrict LLM's knowledge
        system_prompt = """You are an AI assistant that can ONLY answer questions based on the information provided in the vector database and PostgreSQL database. 
        If the question cannot be answered using the available data, respond with: 'Can not answer based on data available in vector DB and postgres DB'.
        Do not use any external knowledge or information."""
        
        # Prepare the user message with context from databases
        user_message = f"Question: {query}\n"
        if api_match:
            user_message += f"Relevant API information: {api_match}\n"
        
        # Get response from LLM
        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ]
        )
        
        self.history.append({'query': query, 'response': response['message']['content']})
        return response['message']['content']

    def get_history(self):
        return self.history
