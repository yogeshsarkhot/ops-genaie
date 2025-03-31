import yaml
import json
import os
from typing import Dict, Any, List
from urllib.parse import urljoin

class OpenAPIParser:
    def __init__(self, openapi_file: str):
        """
        Initialize the parser with the OpenAPI specification
        
        :param openapi_file: Path to the OpenAPI YAML file
        """
        with open(openapi_file, 'r') as file:
            self.spec = yaml.safe_load(file)
    
    def resolve_ref(self, ref: str) -> Dict[str, Any]:
        """
        Resolve JSON reference in OpenAPI specification.
        
        :param ref: Reference string (e.g., '#/components/schemas/SomeSchema')
        :return: Resolved schema
        """
        # Split reference path
        if not ref.startswith('#/'):
            raise ValueError(f"Unsupported reference format: {ref}")
        
        # Remove '#/' and split into components
        ref_parts = ref[2:].split('/')
        
        # Traverse the specification to find the referenced object
        current = self.spec
        for part in ref_parts:
            current = current.get(part, {})
        
        return current
    
    def resolve_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve schema references
        
        :param schema: Schema to resolve
        :return: Fully resolved schema
        """
        # If reference is present, resolve it first
        if '$ref' in schema:
            schema = self.resolve_ref(schema['$ref'])
        
        # Handle allOf references
        if 'allOf' in schema:
            resolved_schema = {}
            for sub_schema in schema['allOf']:
                # Recursively resolve each sub-schema
                sub_resolved = self.resolve_schema(sub_schema)
                
                # Merge properties
                if 'properties' in sub_resolved:
                    resolved_schema.setdefault('properties', {}).update(sub_resolved['properties'])
                
                # Merge other top-level keys
                for key, value in sub_resolved.items():
                    if key != 'properties':
                        resolved_schema[key] = value
            
            return resolved_schema
        
        return schema
    
    def generate_sample_value(self, schema: Dict[str, Any]) -> Any:
        """
        Generate a sample value based on the schema type.
        
        :param schema: OpenAPI schema definition
        :return: Sample value
        """
        # Resolve the schema first
        schema = self.resolve_schema(schema)
        
        schema_type = schema.get('type')
        
        # Handle different schema types
        if schema_type == 'string':
            # Use format or default string
            if schema.get('format') == 'email':
                return 'sample@example.com'
            elif schema.get('format') == 'date':
                return '2024-01-01'
            elif schema.get('format') == 'date-time':
                return '2024-03-26T10:00:00Z'
            return schema.get('example', 'sample_string')
        
        elif schema_type == 'integer':
            return schema.get('example', 1)
        
        elif schema_type == 'number':
            return schema.get('example', 1.0)
        
        elif schema_type == 'boolean':
            return True
        
        elif schema_type == 'array':
            # Generate sample array with one item
            items_schema = schema.get('items', {})
            return [self.generate_sample_value(items_schema)]
        
        elif schema_type == 'object':
            return self.generate_sample_body(schema)
        
        # Default fallback
        return None
    
    def generate_sample_body(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a sample request body based on the schema.
        
        :param schema: OpenAPI schema definition
        :return: Sample request body
        """
        # Resolve the schema first
        schema = self.resolve_schema(schema)
        
        if not schema or schema.get('type') != 'object':
            return None
        
        # Handle object type schemas
        sample_body = {}
        properties = schema.get('properties', {})
        required_props = schema.get('required', [])
        
        for prop_name, prop_schema in properties.items():
            # Resolve property schema
            resolved_prop_schema = self.resolve_schema(prop_schema)
            
            # Generate value for required properties or if value is not None
            if prop_name in required_props or resolved_prop_schema.get('example') is not None:
                sample_body[prop_name] = self.generate_sample_value(resolved_prop_schema)
        
        return sample_body
    
    def generate_sample_inputs(self, output_dir: str = 'api_samples') -> List[Dict[str, Any]]:
        """
        Parse OpenAPI file and generate sample API request inputs.
        
        :param output_dir: Directory to save generated sample request files
        :return: List of sample API request details
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base server URL (use first server if multiple exist)
        base_url = self.spec.get('servers', [{}])[0].get('url', '')
        
        # Prepare output for documentation
        api_samples = []
        
        # Iterate through paths
        for path, path_item in self.spec.get('paths', {}).items():
            # Iterate through HTTP methods for each path
            for method, operation in path_item.items():
                # Skip non-HTTP method keys
                if method in ['parameters', '$ref']:
                    continue
                
                # Construct fully qualified URL
                full_path = urljoin(base_url, path)
                
                # Process URL parameters
                url_params = {}
                path_params = operation.get('parameters', []) + path_item.get('parameters', [])
                for param in path_params:
                    if param.get('in') == 'path':
                        # Generate sample value based on parameter schema
                        url_params[param['name']] = self.generate_sample_value(param.get('schema', {}))
                
                # Process request body
                request_body = None
                if 'requestBody' in operation:
                    # Get request body specification
                    request_body_spec = operation['requestBody']
                    
                    # Get JSON content
                    content = request_body_spec.get('content', {})
                    json_content = content.get('application/json', {})
                    
                    # Get schema, resolving references
                    schema = json_content.get('schema', {})
                    if '$ref' in schema:
                        schema = self.resolve_ref(schema['$ref'])
                    
                    # Generate sample request body
                    request_body = self.generate_sample_body(schema)
                
                # Prepare sample request details
                sample_request = {
                    'method': method.upper(),
                    'full_path': full_path,
                    'summary': operation.get('summary', 'No summary provided'),
                    'description': operation.get('description', 'No description provided'),
                    'url_params': url_params,
                    'request_body': request_body
                }
                api_samples.append(sample_request)
                
                # Save sample request body to file if exists
                if request_body:
                    body_filename = f"{output_dir}/{path.replace('/', '_')}_{method}_request.json"
                    with open(body_filename, 'w') as f:
                        json.dump(request_body, f, indent=2)
        
        return api_samples

# Example usage
if __name__ == '__main__':
    # Parse the OpenAPI specification
    parser = OpenAPIParser('openapi.yaml')
    
    # Generate sample inputs
    samples = parser.generate_sample_inputs()
    
    # Print out generated samples
    for sample in samples:
        print(f"Method: {sample['method']}")
        print(f"Full Path: {sample['full_path']}")
        print(f"Summary: {sample['summary']}")
        print(f"Description: {sample['description']}")
        print(f"URL Params: {sample['url_params']}")
        print(f"Request Body: {sample['request_body']}")
        print("-" * 50)
