import streamlit as st
import tempfile
import chromadb
import ollama
import re
from typing import List, Dict, Any
from APIExecutor import APIExecutor
import requests
import json
import yaml
import os
from urllib.parse import urljoin
#from openapi_parser import OpenAPIParser
#from vector_db import VectorDBHandler

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

class OpenAPIParser:
    def __init__(self, openapi_file):
        """
        Initialize the parser with the OpenAPI specification
        
        :param openapi_file: Path to the OpenAPI YAML file or file-like object
        """
        if isinstance(openapi_file, str):
            with open(openapi_file, 'r') as file:
                self.spec = yaml.safe_load(file)
        else:
            # Assumes file-like object for Streamlit upload
            self.spec = yaml.safe_load(openapi_file)
    
    def resolve_ref(self, ref: str) -> Dict[str, Any]:
        """
        Resolve JSON reference in OpenAPI specification.
        """
        if not ref.startswith('#/'):
            raise ValueError(f"Unsupported reference format: {ref}")
        
        ref_parts = ref[2:].split('/')
        
        current = self.spec
        for part in ref_parts:
            current = current.get(part, {})
        
        return current
    
    def parse_apis(self) -> List[Dict[str, Any]]:
        """
        Parse APIs from OpenAPI specification
        
        :return: List of API descriptions
        """
        apis = []
        base_url = self.spec.get('servers', [{}])[0].get('url', '')
        
        for path, path_item in self.spec.get('paths', {}).items():
            for method, operation in path_item.items():
                if method in ['parameters', '$ref']:
                    continue
                
                full_path = urljoin(base_url, path)
                
                # Unique identifier: method + path
                unique_id = f"{method.upper()} {path}"
                
                api_info = {
                    'unique_id': unique_id,
                    'full_path': full_path,
                    'method': method.upper(),
                    'summary': operation.get('summary', ''),
                    'description': operation.get('description', ''),
                    'request_body_schema': self._extract_request_body_schema(operation),
                    'parameters': operation.get('parameters', [])
                }
                
                apis.append(api_info)
        
        return apis
    
    def _extract_request_body_schema(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract request body schema from operation
        
        :param operation: OpenAPI operation details
        :return: Request body schema
        """
        if 'requestBody' not in operation:
            return {}
        
        request_body = operation['requestBody']
        content = request_body.get('content', {})
        json_content = content.get('application/json', {})
        schema = json_content.get('schema', {})
        
        return schema

class APIExecutor:
    @staticmethod
    def generate_request_details(api_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Generate API request details using Llama3
        
        :param api_info: API information dictionary
        :param query: User's query
        :return: Request details with parameters and body
        """
        print("Debug - Input API Info:")
        print(json.dumps(api_info, indent=2))
        print(f"Debug - User Query: {query}")
        
        prompt = f"""
        API Details:
        {json.dumps(api_info, indent=2)}
        
        User Query: {query}
        
        Generate request parameters and body based on the query. 
        Return a valid JSON with 'url_params' and 'request_body'.
        IMPORTANT: Use ONLY the parameter names from the API path WITHOUT curly braces.
        Example:
        {{
            "url_params": {{
                "account_id": 2
            }},
            "request_body": {{}}
        }}
        """
        
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        print("Debug - LLM Raw Response:")
        print(response['message']['content'])
        
        # Extract JSON from response
        try:
            # Use regex to extract JSON, handling code blocks and markdown
            import re
            json_match = re.search(r'```(?:json)?\n(.*?)```', response['message']['content'], re.DOTALL | re.MULTILINE)
            
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Fallback to finding first JSON-like structure
                json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip()
                else:
                    raise ValueError("No JSON found")
            
            print("Debug - Extracted JSON String:")
            print(json_str)
            
            request_details = json.loads(json_str)
            
            print("Debug - Parsed Request Details:")
            print(json.dumps(request_details, indent=2))
            
            return request_details
        except Exception as e:
            print(f"Debug - JSON Parsing Error: {e}")
            return {}

    @staticmethod
    def execute_api_request(api_info: Dict[str, Any], request_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute API request
        
        :param api_info: API information dictionary
        :param request_details: Request parameters and body
        :return: API response
        """
        print("Debug - Execution Input API Info:")
        print(json.dumps(api_info, indent=2))
        print("Debug - Request Details:")
        print(json.dumps(request_details, indent=2))
        
        method = api_info['method'].lower()
        full_path = api_info['full_path']
        
        print(f"Debug - Original Full Path: {full_path}")
        
        # Extract URL parameter from either API path or request details
        url_param_name = [p.strip('{}') for p in re.findall(r'\{([^{}]*)\}', full_path)][0]
        url_params = request_details.get('url_params', {})
        
        print(f"Debug - Extracted URL Param Name: {url_param_name}")
        print(f"Debug - URL Params: {url_params}")
        
        # Ensure the correct URL parameter is used
        param_value = url_params.get(url_param_name)
        
        if param_value is not None:
            # Replace the parameter in the URL
            full_path = re.sub(r'\{' + url_param_name + r'\}', str(param_value), full_path)
        
        print(f"Debug - Final Full Path: {full_path}")
        
        # Prepare request
        request_kwargs = {
            'url': full_path,
            'method': method
        }
        
        if request_details.get('request_body'):
            request_kwargs['json'] = request_details['request_body']
        
        try:
            print("Debug - Request Kwargs:")
            print(json.dumps(request_kwargs, indent=2))
            
            response = requests.request(**request_kwargs)
            
            print("Debug - Response Status Code:", response.status_code)
            print("Debug - Response Content:", response.text)
            
            return {
                'status_code': response.status_code,
                'body': response.json() if response.content else {}
            }
        except Exception as e:
            print(f"Debug - Exception: {str(e)}")
            return {
                'error': str(e)
            }

def main():
    st.title("API Explorer with OpenAPI, Vector Search, and Execution")
    
    # Sidebar for file upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload OpenAPI YAML", 
        type=['yaml', 'yml']
    )
    
    if uploaded_file is not None:
        # Parse OpenAPI specification
        with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        parser = OpenAPIParser(tmp_path)
        apis = parser.parse_apis()
        
        # Initialize vector database
        vector_db = VectorDBHandler()
        vector_db.add_documents(apis)
        
        # Chat interface
        st.header("API Query Interface")
        
        # Initialize session state for chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask about an API"):
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Find most similar API
            with st.chat_message("assistant"):
                with st.spinner('Finding most relevant API...'):
                    similar_api_id = vector_db.find_most_similar_api(prompt)
                    api_info = next(
                        (api for api in apis if api['unique_id'] == similar_api_id), 
                        None
                    )
                
                if api_info:
                    st.write(f"Most relevant API: {api_info['unique_id']}")
                    st.write(f"Summary: {api_info['summary']}")
                    
                    # Generate request details
                    request_details = APIExecutor.generate_request_details(api_info, prompt)
                    
                    # Execute API request
                    response = APIExecutor.execute_api_request(api_info, request_details)
                    
                    response_text = json.dumps(response, indent=2)
                    st.code(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                else:
                    st.error("No matching API found.")

if __name__ == "__main__":
    main()
