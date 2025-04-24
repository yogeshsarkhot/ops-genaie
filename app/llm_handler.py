import ollama
from app.database import Database
import json
import re
import yaml
import logging

class LLMHandler:
    def __init__(self):
        # Initialize the model and database
        self.model = "llama3-groq-tool-use"
        self.db = Database()
        self._tool_registry = None

    def set_tools(self, tool_registry):
        # Set the tool registry
        self._tool_registry = tool_registry

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
            model="llama3-groq-tool-use",
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
        # Check if the request body schema is empty
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
            model="llama3-groq-tool-use",
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

    def get_response(self, query, tools=None):
        """Get response from LLM and find relevant API using dynamic tools."""
        if tools is not None:
            self.set_tools(tools)
        if not self._tool_registry:
            return "No tools available for query."

        # Gather tool metadata for LLM prompt
        tool_infos = []
        for tool_name, tool_func in self._tool_registry.items():
            api = getattr(tool_func, '__api_metadata__', None)
            if api is None:
                # Fallback: try to get from main's tool_registry
                api = getattr(tool_func, 'api_metadata', None)
            if api is None:
                # As a last resort, try to reconstruct minimal info
                tool_infos.append({
                    'tool_name': tool_name,
                    'method': 'UNKNOWN',
                    'path': 'UNKNOWN',
                    'summary': '',
                    'description': '',
                    'parameters': [],
                    'request_body': None
                })
                continue
            info = {
                'tool_name': tool_name,
                'method': api.get('method'),
                'path': api.get('full_path'),
                'summary': api.get('summary', ''),
                'description': api.get('description', ''),
                'parameters': [p['name'] for p in api.get('parameters', [])],
                'request_body': api.get('request_body') if api.get('request_body') and api.get('request_body') != '{}' else None
            }
            tool_infos.append(info)

        if not tool_infos:
            return f"No tool metadata found. Tool registry: {list(self._tool_registry.keys())}"

        # Build LLM prompt
        prompt = """You are an API assistant. You have access to the following API tools.\n"""
        for tool in tool_infos:
            prompt += f"\nTool Name: {tool['tool_name']}\nMethod: {tool['method']}\nPath: {tool['path']}\nSummary: {tool['summary']}\nDescription: {tool['description']}\nParameters: {tool['parameters']}"
            if tool['request_body']:
                prompt += f"\nRequest Body Schema: {tool['request_body']}"
            prompt += "\n---"
        prompt += f"\n\nUser Query: {query}\n\n"
        prompt += """Based on the user query and available tools, select the best tool to call.\nReturn ONLY a JSON object in the following format:\n{\n  \"tool_name\": <tool_name>,\n  \"parameters\": {<param_name>: <value>, ...},\n  \"request_body\": <dict or null>\n}\nIf no tool matches, return null.\n\nExamples:\nSimple:\n{\n  \"tool_name\": \"GET__accounts\",\n  \"parameters\": {\"skip\": 0, \"limit\": 10},\n  \"request_body\": null\n}\nComplex:\n{\n  \"tool_name\": \"POST__accounts\",\n  \"parameters\": {},\n  \"request_body\": {\n    \"name\": \"Acme Corporation\",\n    \"address_line1\": \"123 Main St\",\n    \"address_line2\": \"Suite 456\",\n    \"city\": \"Anytown\",\n    \"state\": \"CA\",\n    \"zip_code\": \"90210\"\n  }\n}\n\nReturn ONLY the JSON object, and nothing else. Do not include any explanation, Markdown, or extra text.\n"""

        # Call LLM
        response = ollama.chat(
            model="llama3-groq-tool-use",
            messages=[
                {'role': 'system', 'content': 'You are an API assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            options={"temperature": 0.0}
        )
        try:
            content = response['message']['content'].strip()
            logging.info(f"Raw LLM response: {content}")

            # Extract all complete JSON objects using brace counting
            def extract_all_json(text):
                results = []
                i = 0
                while i < len(text):
                    if text[i] == '{':
                        start = i
                        brace_count = 1
                        i += 1
                        while i < len(text) and brace_count > 0:
                            if text[i] == '{':
                                brace_count += 1
                            elif text[i] == '}':
                                brace_count -= 1
                            i += 1
                        if brace_count == 0:
                            results.append(text[start:i])
                        else:
                            break
                    else:
                        i += 1
                return results

            json_candidates = extract_all_json(content)
            logging.info(f"JSON candidates found: {json_candidates}")
            result = None
            json_str = None
            for candidate in json_candidates:
                try:
                    result = json.loads(candidate)
                    json_str = candidate
                    break
                except Exception as parse_exc:
                    logging.info(f"Candidate failed JSON parsing: {candidate} | Error: {parse_exc}")
            if result is None:
                logging.error(f"No valid JSON object found in LLM response. Candidates: {json_candidates}")
                return f"No valid JSON object found in LLM response. The LLM likely returned an explanation or list instead of a JSON object. Please rephrase your query or check the API schema.\nRaw response: {content}"
            logging.info(f"Extracted JSON string: {json_str}")
            if not result:
                return "No matching API tool found for your query."
            tool_name = result.get('tool_name')
            params = result.get('parameters', {})
            request_body = result.get('request_body', None)
            tool_func = self._tool_registry.get(tool_name)
            if tool_func is None:
                logging.error(f"Tool '{tool_name}' not found. Registry: {list(self._tool_registry.keys())}")
                return f"Tool '{tool_name}' not found."
            # Cast parameter types based on API metadata
            api_meta = getattr(tool_func, '__api_metadata__', None)
            if api_meta:
                param_schemas = {p['name']: p.get('schema', {}) for p in api_meta.get('parameters', [])}
                for k, v in params.items():
                    schema = param_schemas.get(k, {})
                    expected_type = schema.get('type')
                    if expected_type == 'integer' and not isinstance(v, int):
                        try:
                            params[k] = int(v)
                            logging.info(f"Casted parameter '{k}' to int: {params[k]}")
                        except Exception as e:
                            logging.warning(f"Failed to cast parameter '{k}' value '{v}' to int: {e}")
                    elif expected_type == 'number' and not isinstance(v, float):
                        try:
                            params[k] = float(v)
                            logging.info(f"Casted parameter '{k}' to float: {params[k]}")
                        except Exception as e:
                            logging.warning(f"Failed to cast parameter '{k}' value '{v}' to float: {e}")
                    elif expected_type == 'boolean' and not isinstance(v, bool):
                        try:
                            if str(v).lower() in ['true', '1', 'yes']:
                                params[k] = True
                            elif str(v).lower() in ['false', '0', 'no']:
                                params[k] = False
                            logging.info(f"Casted parameter '{k}' to bool: {params[k]}")
                        except Exception as e:
                            logging.warning(f"Failed to cast parameter '{k}' value '{v}' to bool: {e}")
            # Call the selected tool
            kwargs = params.copy()
            if request_body:
                kwargs['request_body'] = request_body
            api_result = tool_func(**kwargs)
            logging.info(f"Tool called: {tool_name} | Result: {api_result}")
            return f"Tool called: {tool_name}\nResult: {api_result}"
        except Exception as e:
            logging.error(f"General error in get_response: {e}")
            return f"Error processing LLM response: {str(e)}\nRaw response: {response['message']['content']}"

    def _format_api_response(self, api_data, query):
        # ... (existing code, unchanged)
        pass

    def _extract_parameter_value(self, query, param_name):
        # ... (existing code, unchanged)
        pass

    def _extract_request_body_values(self, query, request_body):
        # ... (existing code, unchanged)
        pass

    def get_history(self):
        # Return query history from database
        return self.db.get_query_history()

    def explain_api_response(self, api_response):
        """Explain API response in plain English."""
        try:
            # Format the response for the LLM
            response_text = f"""API Response:\nStatus Code: {api_response.get('status_code')}\nHeaders: {json.dumps(api_response.get('headers', {}) , indent=2)}\nBody: {json.dumps(api_response.get('body', {}), indent=2)}\n"""
            # Create prompt for explanation
            prompt = f"""Please explain the following API response in plain English. \nFocus on explaining what the response means and any important information it contains.\nKeep the explanation concise but informative.\n\n{response_text}\n\nExplanation:"""
            # Get explanation from LLM
            response = ollama.chat(
                model="llama3.1",
                messages=[
                    {'role': 'system', 'content': 'You are an API response explainer. Explain API responses in clear, plain English.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={"temperature": 0.0}
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error explaining API response: {str(e)}")
            return "Unable to explain the API response."
