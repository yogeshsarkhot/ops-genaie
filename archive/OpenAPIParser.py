import yaml
import json
import os
from typing import Dict, Any, List
from urllib.parse import urljoin

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
