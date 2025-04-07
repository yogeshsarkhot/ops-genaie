import chromadb
import ollama
import json

class EmbeddingHandler:
    def __init__(self, chroma_path="chroma_data"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection("api_embeddings")

    def create_embedding(self, api_data):
        # Create a comprehensive text representation of the API
        text_parts = [
            f"API: {api_data['name']}",
            f"Method: {api_data['method']}",
            f"Summary: {api_data['summary']}",
            f"Description: {api_data['description']}",
            f"Operation ID: {api_data['operation_id']}",
            f"Tags: {', '.join(api_data['tags'])}"
        ]

        # Add parameters
        if api_data['parameters']:
            text_parts.append("Parameters:")
            for param in api_data['parameters']:
                param_text = [
                    f"  - Name: {param['name']}",
                    f"    Location: {param['in']}",
                    f"    Required: {param['required']}",
                    f"    Description: {param['description']}"
                ]
                if param.get('schema'):
                    schema_text = self._format_schema(param['schema'])
                    param_text.append(f"    Schema: {schema_text}")
                text_parts.append('\n'.join(param_text))

        # Add request body
        if api_data['request_body']:
            text_parts.append("Request Body:")
            request_body_text = self._format_schema(api_data['request_body'])
            text_parts.append(request_body_text)

        # Add response schemas
        if api_data['response_schemas']:
            text_parts.append("Responses:")
            for status_code, response in api_data['response_schemas'].items():
                response_text = [
                    f"  Status Code: {status_code}",
                    f"  Description: {response['description']}"
                ]
                if response.get('schema'):
                    schema_text = self._format_schema(response['schema'])
                    response_text.append(f"  Schema: {schema_text}")
                text_parts.append('\n'.join(response_text))

        # Combine all text parts
        text = '\n'.join(text_parts)

        # Create embedding
        embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)['embedding']
        
        # Prepare metadata
        metadata = {
            'name': api_data['name'],
            'method': api_data['method'],
            'summary': api_data['summary'],
            'description': api_data['description'],
            'operation_id': api_data['operation_id'],
            'tags': json.dumps(api_data['tags']),
            'parameters': json.dumps(api_data['parameters']),
            'request_body': json.dumps(api_data['request_body']),
            'response_schemas': json.dumps(api_data['response_schemas']),
            'openapi_version': api_data['openapi_version'],
            'api_title': api_data['api_title'],
            'api_description': api_data['api_description'],
            'api_version': api_data['api_version']
        }
        
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[f"{api_data['name']}_{api_data['method']}"]
        )

    def _format_schema(self, schema):
        """Format schema into a readable text representation."""
        if not schema:
            return "No schema defined"

        if isinstance(schema, str):
            return schema

        if isinstance(schema, dict):
            parts = []
            
            # Handle type
            if 'type' in schema:
                parts.append(f"Type: {schema['type']}")
            
            # Handle required fields
            if 'required' in schema:
                parts.append(f"Required fields: {', '.join(schema['required'])}")
            
            # Handle properties
            if 'properties' in schema:
                parts.append("Properties:")
                for prop_name, prop_value in schema['properties'].items():
                    prop_parts = [f"  - {prop_name}:"]
                    
                    if isinstance(prop_value, dict):
                        if 'type' in prop_value:
                            prop_parts.append(f"    Type: {prop_value['type']}")
                        if 'description' in prop_value:
                            prop_parts.append(f"    Description: {prop_value['description']}")
                        if 'format' in prop_value:
                            prop_parts.append(f"    Format: {prop_value['format']}")
                        if 'minimum' in prop_value:
                            prop_parts.append(f"    Minimum: {prop_value['minimum']}")
                        if 'maximum' in prop_value:
                            prop_parts.append(f"    Maximum: {prop_value['maximum']}")
                        if 'minLength' in prop_value:
                            prop_parts.append(f"    Min Length: {prop_value['minLength']}")
                        if 'maxLength' in prop_value:
                            prop_parts.append(f"    Max Length: {prop_value['maxLength']}")
                        if 'pattern' in prop_value:
                            prop_parts.append(f"    Pattern: {prop_value['pattern']}")
                        if 'enum' in prop_value:
                            prop_parts.append(f"    Enum: {', '.join(str(v) for v in prop_value['enum'])}")
                        if 'items' in prop_value:
                            prop_parts.append(f"    Items: {self._format_schema(prop_value['items'])}")
                    else:
                        prop_parts.append(f"    Value: {prop_value}")
                    
                    parts.append('\n'.join(prop_parts))
            
            return '\n'.join(parts)
        
        return str(schema)

    def search_similar(self, query):
        query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)['embedding']
        results = self.collection.query(query_embeddings=[query_embedding], n_results=1)
        if results['metadatas'] and results['metadatas'][0]:
            metadata = results['metadatas'][0][0]
            # Deserialize JSON fields
            metadata['tags'] = json.loads(metadata['tags'])
            metadata['parameters'] = json.loads(metadata['parameters'])
            metadata['request_body'] = json.loads(metadata['request_body'])
            metadata['response_schemas'] = json.loads(metadata['response_schemas'])
            return metadata
        return None