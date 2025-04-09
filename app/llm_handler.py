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
        """Get response from LLM and find relevant API."""
        try:
            # Get similar API from embeddings
            similar_api = self.embedding_handler.search_similar(query)
            
            if similar_api:
                # Format the response based on the API details
                response = self._format_api_response(similar_api, query)
                return response
            else:
                return "No matching API found for your query."
                
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return f"Error processing your query: {str(e)}"

    def _format_api_response(self, api_data, query):
        """Format the API response in the requested format."""
        try:
            # Extract base path from full path
            base_path = api_data['full_path'].split(api_data['base_url'])[1]
            
            # Format the response
            response_lines = [
                f"- API Base Path: {base_path}",
                f"- API Summary: {api_data['summary']}",
                f"- API Description: {api_data['description']}",
                f"- API Method: {api_data['method']}",
                f"- API Full Path: {api_data['full_path']}"
            ]
            
            # Handle URL parameters
            full_path_with_params = api_data['full_path']
            if api_data['parameters']:
                for param in api_data['parameters']:
                    if param['in'] == 'path':
                        # Extract parameter value from query
                        param_value = self._extract_parameter_value(query, param['name'])
                        if param_value:
                            # Replace the parameter in the URL
                            param_pattern = f"{{{param['name']}}}"
                            full_path_with_params = full_path_with_params.replace(param_pattern, str(param_value))
            
            response_lines.append(f"- API Full Path with parameters: {full_path_with_params}")
            
            # Handle request body
            request_body = "None"
            if api_data['request_body']:
                try:
                    # Parse the YAML request body
                    request_body_dict = yaml.safe_load(api_data['request_body'])
                    if request_body_dict:
                        # Extract values from query
                        request_body_dict = self._extract_request_body_values(query, request_body_dict)
                        request_body = json.dumps(request_body_dict, indent=2)
                except (yaml.YAMLError, json.JSONDecodeError):
                    request_body = api_data['request_body']
            
            response_lines.append(f"- API Request Body: {request_body}")
            
            return "\n".join(response_lines)
            
        except Exception as e:
            print(f"Error formatting API response: {str(e)}")
            return str(api_data)

    def _extract_parameter_value(self, query, param_name):
        """Extract parameter value from user query."""
        try:
            # Convert param_name to a more natural form for matching
            natural_name = param_name.replace('_', ' ').lower()
            
            # Look for patterns like "ID 3" or "policy ID 3"
            pattern = rf"{natural_name}\s+(\d+)"
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
            
            # Look for patterns like "policy_id=3" or "policy-id=3"
            pattern = rf"{param_name}[=:]\s*(\d+)"
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
            
            # Look for patterns like "policy 3" or "policy with ID 3"
            pattern = rf"{natural_name.split()[-1]}\s+(\d+)"
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
            
            # Look for patterns like "ID: 3" or "ID is 3"
            pattern = rf"{natural_name}\s*[:=]\s*(\d+)"
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
            
            return None
        except Exception as e:
            print(f"Error extracting parameter value: {str(e)}")
            return None

    def _extract_request_body_values(self, query, request_body):
        """Extract request body values from user query."""
        try:
            # Convert the request body to a flat dictionary for easier processing
            flat_body = {}
            self._flatten_dict(request_body, flat_body)
            
            # Extract values for each field
            for field, value in flat_body.items():
                if isinstance(value, str):
                    # Look for the field name in the query
                    natural_name = field.replace('_', ' ').lower()
                    pattern = rf"{natural_name}\s+([^,]+)"
                    match = re.search(pattern, query.lower())
                    if match:
                        flat_body[field] = match.group(1).strip()
            
            # Reconstruct the nested structure
            return self._unflatten_dict(flat_body)
        except Exception as e:
            print(f"Error extracting request body values: {str(e)}")
            return request_body

    def _flatten_dict(self, d, flat_dict, prefix=''):
        """Flatten a nested dictionary."""
        for k, v in d.items():
            if isinstance(v, dict):
                self._flatten_dict(v, flat_dict, f"{prefix}{k}.")
            else:
                flat_dict[f"{prefix}{k}"] = v

    def _unflatten_dict(self, flat_dict):
        """Convert a flat dictionary back to a nested structure."""
        result = {}
        for k, v in flat_dict.items():
            parts = k.split('.')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = v
        return result

    def get_history(self):
        return self.db.get_query_history()
