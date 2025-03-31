import requests
import ollama
import json
from typing import Dict, Any

class APIExecutor:
    @staticmethod
    def generate_request_details(api_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Generate API request details using Llama3
        
        :param api_info: API information dictionary
        :param query: User's query
        :return: Request details with parameters and body
        """
        prompt = f"""
        API Details:
        {json.dumps(api_info, indent=2)}
        
        User Query: {query}
        
        Generate request parameters and body based on the query. 
        Return a JSON with 'url_params' and 'request_body'.
        If no specific values can be inferred, use example values from schema.
        """
        
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        try:
            request_details = json.loads(response['message']['content'])
            return request_details
        except json.JSONDecodeError:
            return {}
    
    @staticmethod
    def execute_api_request(api_info: Dict[str, Any], request_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute API request
        
        :param api_info: API information dictionary
        :param request_details: Request parameters and body
        :return: API response
        """
        method = api_info['method'].lower()
        full_path = api_info['full_path']
        
        # Replace URL parameters
        for param, value in request_details.get('url_params', {}).items():
            full_path = full_path.replace(f"{{{param}}}", str(value))
        
        # Prepare request
        request_kwargs = {
            'url': full_path,
            'method': method
        }
        
        if request_details.get('request_body'):
            request_kwargs['json'] = request_details['request_body']
        
        try:
            response = requests.request(**request_kwargs)
            return {
                'status_code': response.status_code,
                'body': response.json() if response.content else {}
            }
        except Exception as e:
            return {
                'error': str(e)
            }
