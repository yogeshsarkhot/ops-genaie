import yaml
import os
from datetime import datetime

class FileProcessor:
    def __init__(self, upload_dir="uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    def process_uploaded_file(self, uploaded_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(self.upload_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        return self.parse_yaml(file_path), filename

    def resolve_schema_reference(self, ref, components):
        """Resolve a schema reference recursively."""
        if not ref or not isinstance(ref, str) or not ref.startswith('#/components/schemas/'):
            return ref

        schema_name = ref.split('/')[-1]
        schema = components.get('schemas', {}).get(schema_name, {})
        
        if not schema:
            return ref

        # Handle allOf references
        if 'allOf' in schema:
            resolved_schema = {}
            for item in schema['allOf']:
                if isinstance(item, dict) and '$ref' in item:
                    ref_schema = self.resolve_schema_reference(item['$ref'], components)
                    if isinstance(ref_schema, dict):
                        resolved_schema.update(ref_schema)
                elif isinstance(item, dict):
                    resolved_schema.update(item)
            return resolved_schema

        # Handle direct schema definitions
        resolved_schema = {}
        for key, value in schema.items():
            if key == 'properties':
                resolved_schema['properties'] = {}
                for prop_name, prop_value in value.items():
                    if isinstance(prop_value, dict) and '$ref' in prop_value:
                        resolved_schema['properties'][prop_name] = self.resolve_schema_reference(prop_value['$ref'], components)
                    else:
                        resolved_schema['properties'][prop_name] = prop_value
            elif isinstance(value, dict) and '$ref' in value:
                resolved_schema[key] = self.resolve_schema_reference(value['$ref'], components)
            else:
                resolved_schema[key] = value

        return resolved_schema

    def resolve_response_schema(self, schema, components):
        """Resolve response schema references recursively."""
        if not schema:
            return schema

        if isinstance(schema, dict):
            resolved = {}
            for key, value in schema.items():
                if key == '$ref':
                    return self.resolve_schema_reference(value, components)
                elif key == 'items' and isinstance(value, dict):
                    resolved[key] = self.resolve_response_schema(value, components)
                else:
                    resolved[key] = value
            return resolved
        elif isinstance(schema, list):
            return [self.resolve_response_schema(item, components) for item in schema]
        
        return schema

    def parse_yaml(self, file_path):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError("Invalid YAML: File must contain a dictionary")

        # Extract API information
        apis = []
        
        # Get OpenAPI version and info
        openapi_version = data.get('openapi', '')
        info = data.get('info', {})
        title = info.get('title', '')
        description = info.get('description', '')
        version = info.get('version', '')
        
        # Get components for schema resolution
        components = data.get('components', {})
        
        # Process paths
        for path, path_data in data.get('paths', {}).items():
            if not isinstance(path_data, dict):
                continue

            # Extract path-level parameters
            path_parameters = []
            for param in path_data.get('parameters', []):
                if isinstance(param, dict):
                    param_dict = {
                        'name': param.get('name', ''),
                        'in': param.get('in', ''),
                        'description': param.get('description', ''),
                        'required': param.get('required', False),
                        'schema': param.get('schema', {})
                    }
                    path_parameters.append(param_dict)

            for method, method_data in path_data.items():
                if method.lower() not in ['get', 'post', 'put', 'delete', 'patch']:
                    continue

                # Handle case where method_data might be a list
                if isinstance(method_data, list):
                    method_data = next((item for item in method_data if isinstance(item, dict)), {})
                elif not isinstance(method_data, dict):
                    continue

                # Extract operation details
                operation_id = method_data.get('operationId', '')
                summary = method_data.get('summary', '')
                description = method_data.get('description', '')
                tags = method_data.get('tags', [])
                
                # Extract operation-level parameters
                operation_parameters = []
                for param in method_data.get('parameters', []):
                    if isinstance(param, dict):
                        param_dict = {
                            'name': param.get('name', ''),
                            'in': param.get('in', ''),
                            'description': param.get('description', ''),
                            'required': param.get('required', False),
                            'schema': param.get('schema', {})
                        }
                        operation_parameters.append(param_dict)

                # Combine path and operation parameters
                all_parameters = path_parameters.copy()
                for op_param in operation_parameters:
                    existing_param = next(
                        (p for p in all_parameters if p['name'] == op_param['name'] and p['in'] == op_param['in']),
                        None
                    )
                    if existing_param:
                        existing_param.update(op_param)
                    else:
                        all_parameters.append(op_param)

                # Extract request body
                request_body = {}
                if 'requestBody' in method_data:
                    content = method_data['requestBody'].get('content', {})
                    if 'application/json' in content:
                        request_body = content['application/json'].get('schema', {})
                        if isinstance(request_body, dict) and '$ref' in request_body:
                            request_body = self.resolve_schema_reference(request_body['$ref'], components)

                # Extract and resolve response schemas
                response_schemas = {}
                for status_code, response_data in method_data.get('responses', {}).items():
                    if isinstance(response_data, dict):
                        content = response_data.get('content', {})
                        schema = {}
                        if 'application/json' in content:
                            schema = content['application/json'].get('schema', {})
                            schema = self.resolve_response_schema(schema, components)
                        response_schemas[status_code] = {
                            'description': response_data.get('description', ''),
                            'schema': schema
                        }

                # Create API dictionary
                api_dict = {
                    'name': path,
                    'method': method.upper(),
                    'operation_id': operation_id,
                    'summary': summary,
                    'description': description,
                    'tags': tags,
                    'parameters': all_parameters,
                    'request_body': request_body,
                    'response_schemas': response_schemas,
                    'openapi_version': openapi_version,
                    'api_title': title,
                    'api_description': description,
                    'api_version': version
                }
                
                apis.append(api_dict)
        
        return apis