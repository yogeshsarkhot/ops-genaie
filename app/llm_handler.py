import ollama
from app.database import Database
from app.embeddings import EmbeddingHandler
import json
import re
import yaml

class LLMHandler:
    def __init__(self):
        self.model = "llama3"
        self.db = Database()
        self.embedding_handler = EmbeddingHandler()

    def _replace_url_parameters(self, full_path, query):
        """
        Replace URL parameters in the full path with values extracted from the query.
        Returns the modified path and any extracted parameters.
        """
        # Extract URL parameters using regex
        url_param_matches = re.findall(r'\{([^{}]*)\}', full_path)
        if not url_param_matches:
            return full_path, {}

        # Create a prompt to extract parameter values from the query
        param_prompt = f"""Given the following query and URL parameters, extract the values for each parameter.
        
Query: {query}
URL Parameters: {', '.join(url_param_matches)}

Return a JSON object with parameter names and their values. If a value cannot be determined, use null.
Example format:
{{
    "parameter_name": "extracted_value"
}}
"""
        # Get parameter values from LLM
        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are a parameter extractor. Extract values for URL parameters from the given query.'},
                {'role': 'user', 'content': param_prompt}
            ],
            options={"temperature": 0.0}
        )

        try:
            param_values = json.loads(response['message']['content'])
        except json.JSONDecodeError:
            param_values = {}

        # Replace parameters in the URL
        modified_path = full_path
        for param_name in url_param_matches:
            clean_param_name = param_name.strip('{}')
            param_value = param_values.get(clean_param_name)
            if param_value is not None:
                modified_path = re.sub(r'\{' + param_name + r'\}', str(param_value), modified_path)

        return modified_path, param_values

    def _extract_request_body(self, request_body_schema, query):
        """
        Extract request body values from the query based on the schema.
        Returns the populated request body JSON.
        """
        if not request_body_schema:
            return {}

        try:
            # Parse YAML schema to dict
            schema_dict = yaml.safe_load(request_body_schema)
        except yaml.YAMLError:
            return {}

        # Create a prompt to extract request body values
        schema_prompt = f"""Given the following query and request body schema, extract the values for each field.
        
Query: {query}
Request Body Schema:
{yaml.dump(schema_dict, default_flow_style=False)}

Return a JSON object with field names and their values. Only include fields that have values in the query.
Example format:
{{
    "field_name": "extracted_value"
}}
"""
        # Get request body values from LLM
        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are a request body extractor. Extract values for request body fields from the given query.'},
                {'role': 'user', 'content': schema_prompt}
            ],
            options={"temperature": 0.0}
        )

        try:
            return json.loads(response['message']['content'])
        except json.JSONDecodeError:
            return {}

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
            # Replace URL parameters if present
            full_path = api_match['full_path']
            modified_path, param_values = self._replace_url_parameters(full_path, query)
            
            # Extract request body if present
            request_body = {}
            if api_match.get('request_body'):
                request_body = self._extract_request_body(api_match['request_body'], query)
            
            user_message += f"Relevant API information:\n"
            user_message += f"- API: {api_match['name']}\n"
            user_message += f"- Method: {api_match['method']}\n"
            user_message += f"- Full URL: {modified_path}\n"
            if param_values:
                user_message += f"- URL Parameters: {json.dumps(param_values, indent=2)}\n"
            if request_body:
                user_message += f"- Request Body: {json.dumps(request_body, indent=2)}\n"
            user_message += f"- Summary: {api_match['summary']}\n"
            user_message += f"- Description: {api_match['description']}\n"
        
        # Get response from LLM
        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            options={"temperature": 0.0}
        )
        
        # Store query and response in database
        self.db.save_query_history(query, response['message']['content'])
        return response['message']['content']

    def get_history(self):
        return self.db.get_query_history()
