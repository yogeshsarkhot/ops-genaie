import ollama
from app.database import Database
from app.embeddings import EmbeddingHandler
import json
import re
import yaml

class LLMHandler:
    def __init__(self):
        #self.model = "llama3"
        self.model = "llama3-groq-tool-use"
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
                try:
                    # Format the response based on the API details
                    response = self._format_api_response(similar_api, query)
                    return response
                except Exception as e:
                    print(f"Error formatting API response: {str(e)}")
                    return f"Error processing API response: {str(e)}"
            else:
                # If no similar API found, try to find a relevant API using LLM
                prompt = f"""Given the following query, find the most relevant API endpoint from the available APIs.
                
Query: {query}

Return a JSON object with the following structure:
{{
    "api_name": "name of the API",
    "method": "HTTP method",
    "description": "brief description of what the API does"
}}

If no relevant API is found, return null.
"""
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': 'You are an API finder. Find the most relevant API endpoint for the given query.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    options={"temperature": 0.0}
                )
                
                try:
                    api_info = json.loads(response['message']['content'])
                    if api_info:
                        # Try to find the API in the database
                        with self.db.conn.cursor() as cur:
                            cur.execute(
                                """SELECT * FROM api_data 
                                WHERE name ILIKE %s 
                                OR description ILIKE %s 
                                OR summary ILIKE %s 
                                LIMIT 1""",
                                (f"%{api_info['api_name']}%", 
                                 f"%{api_info['description']}%",
                                 f"%{api_info['description']}%")
                            )
                            api_data = cur.fetchone()
                            
                            if api_data:
                                # Convert the database row to a dictionary
                                api_dict = {
                                    'name': api_data[2],
                                    'method': api_data[3],
                                    'summary': api_data[4],
                                    'description': api_data[5],
                                    'parameters': json.loads(api_data[9]),
                                    'request_body': api_data[10],
                                    'response_schemas': json.loads(api_data[11]),
                                    'base_url': api_data[7],
                                    'full_path': api_data[8],
                                    'operation_id': api_data[6],
                                    'tags': json.loads(api_data[12]),
                                    'openapi_version': api_data[13],
                                    'api_title': api_data[14],
                                    'api_description': api_data[15],
                                    'api_version': api_data[16]
                                }
                                return self._format_api_response(api_dict, query)
                    
                    return "No matching API found for your query. Please try rephrasing your query or check if the API documentation has been uploaded."
                except json.JSONDecodeError:
                    return "Error processing your query. Please try again."
                
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
                        # Store the request body as a dictionary for API calls
                        api_data['request_body_dict'] = request_body_dict
                        # Format the request body for display
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
            # If request_body is a schema, extract the properties
            if isinstance(request_body, dict) and 'properties' in request_body:
                properties = request_body['properties']
            else:
                properties = request_body

            # Create a more detailed prompt for the LLM
            prompt = f"""Given the following user query and request body schema, extract the values for each field.
            
User Query: {query}

Request Body Schema:
{json.dumps(properties, indent=2)}

Your task is to:

1. Read the user query carefully
2. Identify any values mentioned in the query that match the fields in the schema
3. Return a JSON object with the fields that have values in the query
4. Use the exact field names from the schema
5. Convert any values to the appropriate type (string, number, boolean)

Example:
If the schema has fields "name", "address_line1", "address_line2", "city", "state", "zip_code" and the query is "Create an account for Neo Corp located at 250 Main Street, Suite 250, Avon, CT, 06001",
you should return:
{{{{
  "name": "Neo Corp",
  "address_line1": "250 Main Street",
  "address_line2": "Suite 250",
  "city": "Avon",
  "state": "CT",
  "zip_code": "06001"
}}}}

Return ONLY the JSON object with the extracted values. Do not include any explanations or additional text.
"""
            # Get values from LLM
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a request body extractor. Extract values for request body fields from the given query.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={"temperature": 0.0}
            )

            try:
                # Extract the JSON content from the response
                content = response['message']['content'].strip()
                
                # Remove any markdown code block markers if present
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # Parse the JSON response
                extracted_values = json.loads(content)
                
                # Create a clean request body with only the extracted values
                clean_request_body = {}
                for field, value in extracted_values.items():
                    if value is not None:  # Only include fields with values
                        clean_request_body[field] = value
                
                return clean_request_body
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response for request body values: {str(e)}")
                print(f"LLM Response: {response['message']['content']}")
                return {}
                
        except Exception as e:
            print(f"Error extracting request body values: {str(e)}")
            return {}

    def get_history(self):
        return self.db.get_query_history()

    def summarize_api_response(self, api_response):
        """Summarize API response in plain English."""
        try:
            # Format the response for the LLM
            response_text = f"""API Response:
Status Code: {api_response.get('status_code')}
Headers: {json.dumps(api_response.get('headers', {}), indent=2)}
Body: {json.dumps(api_response.get('body', {}), indent=2)}
"""
            
            # Create prompt for summarization
            prompt = f"""Please summarize the following API response in plain English. 
Focus on explaining what the response means and any important information it contains.
Keep the summary concise but informative.

{response_text}

Summary:"""
            
            # Get summary from LLM
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are an API response summarizer. Explain API responses in clear, plain English.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={"temperature": 0.0}
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            print(f"Error summarizing API response: {str(e)}")
            return "Unable to summarize the API response."
