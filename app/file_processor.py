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
        
        # Get server URL
        servers = data.get('servers', [])
        base_url = servers[0].get('url', '') if servers else ''
        
        # Get components for schema resolution
        components = data.get('components', {})
        
        # Process paths
        paths = data.get('paths', {})
        for path, path_item in paths.items():
            for method, method_data in path_item.items():
                # Skip if method_data is not a dict (e.g., if it's a list or something else)
                if not isinstance(method_data, dict):
                    continue
                operation_id = method_data.get('operationId', None)
                summary = method_data.get('summary', '')
                description = method_data.get('description', '')
                tags = method_data.get('tags', [])

                # Extract parameters
                path_parameters = path_item.get('parameters', [])
                operation_parameters = method_data.get('parameters', [])
                operation_parameters = operation_parameters if isinstance(operation_parameters, list) else []
                path_parameters = path_parameters if isinstance(path_parameters, list) else []
                # Normalize parameter dicts
                def normalize_param(param):
                    return {
                        'name': param.get('name', ''),
                        'in': param.get('in', ''),
                        'description': param.get('description', ''),
                        'required': param.get('required', False),
                        'schema': param.get('schema', {})
                    }
                operation_parameters = [normalize_param(p) for p in operation_parameters]
                path_parameters = [normalize_param(p) for p in path_parameters]

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
                        # Convert request body to YAML string
                        request_body = yaml.dump(request_body, default_flow_style=False, sort_keys=False)
                    elif 'application/yaml' in content:
                        request_body = content['application/yaml'].get('schema', {})
                        if isinstance(request_body, dict) and '$ref' in request_body:
                            request_body = self.resolve_schema_reference(request_body['$ref'], components)
                        # Keep as YAML string
                        request_body = yaml.dump(request_body, default_flow_style=False, sort_keys=False)

                # Extract and resolve response schemas
                response_schemas = {}
                for status_code, response_data in method_data.get('responses', {}).items():
                    if isinstance(response_data, dict):
                        content = response_data.get('content', {})
                        schema = {}
                        if 'application/json' in content:
                            schema = content['application/json'].get('schema', {})
                            schema = self.resolve_response_schema(schema, components)
                            # Convert response schema to YAML string
                            schema = yaml.dump(schema, default_flow_style=False)
                        response_schemas[status_code] = {
                            'description': response_data.get('description', ''),
                            'schema': schema
                        }

                # Create full API path by combining base URL and path
                full_path = f"{base_url.rstrip('/')}/{path.lstrip('/')}" if base_url else path

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
                    'api_version': version,
                    'base_url': base_url,
                    'full_path': full_path
                }
                
                apis.append(api_dict)
        
        return apis