import chromadb
import ollama
import json
import yaml
import os

class EmbeddingHandler:
    def __init__(self, chroma_path="chroma_data"):
        # Ensure the directory exists
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize ChromaDB with explicit settings
        settings = chromadb.config.Settings(
            is_persistent=True,
            persist_directory=chroma_path,
            anonymized_telemetry=False
        )
        
        try:
            self.client = chromadb.PersistentClient(
                path=chroma_path,
                settings=settings
            )
            self.collection = self.client.get_or_create_collection(
                name="api_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChromaDB: {str(e)}")
            # Create a dummy client that will fail gracefully
            self.client = None
            self.collection = None

    def create_embedding(self, api_data):
        # Create a comprehensive text representation of the API
        text_parts = [
            f"API: {api_data['name']}",
            f"Method: {api_data['method']}",
            f"Full URL: {api_data['full_path']}",
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
            # Parse YAML string back to dict for formatting
            try:
                request_body_dict = yaml.safe_load(api_data['request_body'])
                request_body_text = self._format_schema(request_body_dict)
                text_parts.append(request_body_text)
            except yaml.YAMLError:
                text_parts.append(api_data['request_body'])

        # Add response schemas
        if api_data['response_schemas']:
            text_parts.append("Responses:")
            for status_code, response in api_data['response_schemas'].items():
                response_text = [
                    f"  Status Code: {status_code}",
                    f"  Description: {response['description']}"
                ]
                if response.get('schema'):
                    # Parse YAML string back to dict for formatting
                    try:
                        schema_dict = yaml.safe_load(response['schema'])
                        schema_text = self._format_schema(schema_dict)
                        response_text.append(f"  Schema: {schema_text}")
                    except yaml.YAMLError:
                        response_text.append(f"  Schema: {response['schema']}")
                text_parts.append('\n'.join(response_text))

        # Combine all text parts
        text = '\n'.join(text_parts)

        # Create embedding
        embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)['embedding']
        
        # Prepare metadata - convert complex types to JSON strings
        metadata = {
            'name': str(api_data['name']),
            'method': str(api_data['method']),
            'full_path': str(api_data['full_path']),
            'base_url': str(api_data['base_url']),
            'summary': str(api_data['summary']),
            'description': str(api_data['description']),
            'operation_id': str(api_data['operation_id']),
            'tags': json.dumps(api_data['tags']),  # Convert list to JSON string
            'parameters': json.dumps(api_data['parameters']),  # Convert list to JSON string
            'request_body': str(api_data['request_body']),  # Already a string
            'response_schemas': json.dumps(api_data['response_schemas']),  # Convert dict to JSON string
            'openapi_version': str(api_data['openapi_version']),
            'api_title': str(api_data['api_title']),
            'api_description': str(api_data['api_description']),
            'api_version': str(api_data['api_version'])
        }
        
        # Only add to collection if client is initialized
        if self.client and self.collection:
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
        if not self.client or not self.collection:
            return None
            
        query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)['embedding']
        results = self.collection.query(query_embeddings=[query_embedding], n_results=1)
        
        if results['metadatas'] and results['metadatas'][0]:
            metadata = results['metadatas'][0][0]
            try:
                # Deserialize JSON fields
                metadata['tags'] = json.loads(metadata['tags'])
                metadata['parameters'] = json.loads(metadata['parameters'])
                metadata['response_schemas'] = json.loads(metadata['response_schemas'])
                return metadata
            except json.JSONDecodeError as e:
                print(f"Error deserializing metadata: {e}")
                return None
        return None