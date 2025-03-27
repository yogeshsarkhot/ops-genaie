import streamlit as st
import ollama
import chromadb
import os
import requests
import json
import uuid
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('api_chatbot.log')  # Optional: Also log to file
    ]
)
logger = logging.getLogger(__name__)

class OllamaEmbedding:
    def __init__(self, model_name="nomic-embed-text", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        logger.info(f"Initializing OllamaEmbedding with model: {model_name}")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for given input using Ollama
        
        Args:
            input (List[str]): List of strings to embed
        
        Returns:
            List[List[float]]: List of embeddings
        """
        logger.info(f"Generating embeddings for {len(input)} input texts")
        embeddings = []
        for text in input:
            try:
                # Use Ollama API directly to generate embeddings
                logger.debug(f"Generating embedding for text: {text[:50]}...")
                response = requests.post(
                    f"{self.host}/api/embeddings", 
                    json={
                        "model": self.model_name,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                embedding = response.json()['embedding']
                embeddings.append(embedding)
                logger.debug(f"Successfully generated embedding for text")
            except Exception as e:
                logger.error(f"Embedding generation error: {e}")
                embeddings.append([0] * 768)  # Default embedding if generation fails
        return embeddings

class APIChatbot:
    def __init__(self):
        logger.info("Initializing APIChatbot")
        # Streamlit page configuration
        st.set_page_config(page_title="API Definition Chatbot", page_icon="🤖")
        
        # Ensure ChromaDB persistence directory exists
        persist_directory = "./.chromadb/"
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(f"ChromaDB persistence directory: {persist_directory}")
        
        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        logger.info("ChromaDB client initialized")
        
        # Custom Ollama Embedding Function
        self.embedding_function = OllamaEmbedding(
            model_name="nomic-embed-text"
        )
        
        # Create or get collection with custom embedding
        self.collection = self.chroma_client.get_or_create_collection(
            name="api_definitions",
            embedding_function=self.embedding_function
        )
        logger.info("ChromaDB collection 'api_definitions' created/retrieved")
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            logger.info("Initialized empty chat history")
        
        # Initialize session state for API definitions
        if 'api_definitions' not in st.session_state:
            st.session_state.api_definitions = {}
            logger.info("Initialized empty API definitions")

    def add_api_definition(self, swagger_text):
        """
        Parse Swagger text and store API definitions in ChromaDB
        """
        logger.info("Starting to add API definition")
        try:
            # Parse Swagger text (simplified example)
            swagger_data = json.loads(swagger_text)
            logger.info(f"Successfully parsed Swagger JSON with {len(swagger_data.get('paths', {}))} paths")
            
            added_definitions = 0
            for path, path_details in swagger_data.get('paths', {}).items():
                for method, operation in path_details.items():
                    # Create a unique ID for each API definition
                    doc_id = str(uuid.uuid4())
                    
                    # Prepare metadata
                    metadata = {
                        'path': path,
                        'method': method,
                        'description': operation.get('description', ''),
                        'summary': operation.get('summary', '')
                    }
                    
                    # Prepare document for embedding
                    # Combine relevant text for more meaningful embedding
                    document_text = f"{path} {method} {metadata['description']} {metadata['summary']}"
                    
                    # Store in ChromaDB
                    self.collection.add(
                        documents=[document_text],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
                    logger.debug(f"Added API definition: {path} - {method}")
                    
                    # Store in session state for easier access
                    st.session_state.api_definitions[doc_id] = {
                        'path': path,
                        'method': method,
                        'details': operation
                    }
                    added_definitions += 1
            
            logger.info(f"Added {added_definitions} API definitions")
            st.success(f"Added {added_definitions} API definitions")
        except Exception as e:
            logger.error(f"Error parsing API definition: {e}")
            st.error(f"Error parsing API definition: {e}")

    def find_most_relevant_api(self, user_query):
        """
        Use Nomic embedding to find most relevant API definition
        """
        logger.info(f"Finding most relevant API for query: {user_query}")
        # Query the collection to find most similar API definitions
        results = self.collection.query(
            query_texts=[user_query],
            n_results=3  # Top 3 most similar API definitions
        )
        
        # If we have results, return the first (most relevant) API ID
        if results['ids'] and results['ids'][0]:
            most_relevant_id = results['ids'][0][0]
            logger.info(f"Most relevant API ID found: {most_relevant_id}")
            return most_relevant_id
        
        logger.warning("No relevant API found for the query")
        return None

    def generate_api_request(self, user_query):
        """
        Use Ollama Llama3 to generate appropriate API request details
        """
        logger.info(f"Generating API request for query: {user_query}")
        # First, find the most relevant API definition
        api_id = self.find_most_relevant_api(user_query)
        
        if not api_id:
            logger.error("No matching API found for the query")
            st.error("No matching API found for the query")
            return None
        
        # Retrieve the API definition
        api_def = st.session_state.api_definitions.get(api_id)
        
        if not api_def:
            logger.error("API definition retrieval failed")
            st.error("API definition retrieval failed")
            return None
        
        # Use Llama3 to generate API request details
        prompt = f"""
        API Context:
        Path: {api_def['path']}
        Method: {api_def['method']}
        Details: {json.dumps(api_def['details'])}

        User Query: "{user_query}"

        Generate an appropriate API request with:
        - selected_api_id: {api_id}
        - request_body: JSON body for the request (if applicable)
        - path_params: Dictionary of path parameters (if applicable)
        - query_params: Dictionary of query parameters (if applicable)

        Respond in strict JSON format.
        """
        
        logger.info("Calling Ollama Llama3 to generate API request")
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': prompt}
        ])
        logger.info("Response from Ollama to generate API request: {}", response)
        
        try:
            api_request = json.loads(response['message']['content'])
            api_request['selected_api_id'] = api_id
            logger.info("Successfully generated API request")
            return api_request
        except Exception as e:
            logger.error(f"Error generating API request: {e}")
            st.error(f"Error generating API request: {e}")
            return None

    def execute_api_request(self, api_request):
        """
        Execute the API request based on generated details
        """
        logger.info("Attempting to execute API request")
        if not api_request or 'selected_api_id' not in api_request:
            logger.error("Invalid API request")
            st.error("Invalid API request")
            return None
        
        # Retrieve API definition
        api_def = st.session_state.api_definitions.get(api_request['selected_api_id'])
        
        if not api_def:
            logger.error("API definition not found")
            st.error("API definition not found")
            return None
        
        # Construct full URL and request
        base_url = "http://localhost:8000"  # Replace with actual base URL
        full_url = base_url + api_def['path']
        logger.info(f"Constructed URL: {full_url}")
        
        # Replace path parameters
        for param, value in api_request.get('path_params', {}).items():
            full_url = full_url.replace(f"{{{param}}}", str(value))
            logger.debug(f"Replaced path parameter {param} with {value}")
        
        # Prepare request
        request_method = api_def['method'].lower()
        request_kwargs = {
            'url': full_url,
            'params': api_request.get('query_params', {}),
            'json': api_request.get('request_body', {})
        }
        
        # Execute request
        try:
            logger.info(f"Executing {request_method.upper()} request")
            if request_method == 'get':
                response = requests.get(**request_kwargs)
            elif request_method == 'post':
                response = requests.post(**request_kwargs)
            elif request_method == 'put':
                response = requests.put(**request_kwargs)
            elif request_method == 'delete':
                response = requests.delete(**request_kwargs)
            else:
                logger.error(f"Unsupported HTTP method: {request_method}")
                st.error(f"Unsupported HTTP method: {request_method}")
                return None
            
            response_json = response.json()
            logger.info("API request successful")
            return response_json
        except Exception as e:
            logger.error(f"API request failed: {e}")
            st.error(f"API request failed: {e}")
            return None

    def chat_interface(self):
        """
        Streamlit chat interface
        """
        logger.info("Initializing chat interface")
        st.title("API Definition Chatbot 🤖")
        
        # Swagger/API Definition Input
        st.sidebar.header("Add API Definitions")
        swagger_text = st.sidebar.text_area("Paste Swagger JSON")
        if st.sidebar.button("Add API Definitions"):
            self.add_api_definition(swagger_text)
        
        # Chat Interface
        st.header("Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        
        # User input
        if prompt := st.chat_input("Enter your query"):
            logger.info(f"Received user query: {prompt}")
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user', 
                'content': prompt
            })
            
            # Generate API request
            api_request = self.generate_api_request(prompt)
            
            # Execute API request
            api_response = self.execute_api_request(api_request)
            
            # Prepare response using Ollama Llama3
            if api_response:
                logger.info("Generating human-friendly explanation")
                llm_response = ollama.chat(model='llama3', messages=[
                    {
                        'role': 'user', 
                        'content': f"User query: {prompt}\nAPI Response: {json.dumps(api_response)}\n\nProvide a human-friendly explanation of the API response."
                    }
                ])
                
                # Add AI response to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant', 
                    'content': llm_response['message']['content']
                })
                logger.info("Response added to chat history")
            
            # Rerun to update chat interface
            st.rerun()

def main():
    logger.info("Starting API Chatbot application")
    chatbot = APIChatbot()
    chatbot.chat_interface()

if __name__ == "__main__":
    main()
