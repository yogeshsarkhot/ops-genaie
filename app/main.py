import streamlit as st
from app.database import Database
from app.file_processor import FileProcessor
from app.embeddings import EmbeddingHandler
from app.llm_handler import LLMHandler
import json
import requests

def main():
    st.set_page_config(page_title="API Analyzer", layout="wide")
    
    # Initialize components
    db = Database()
    db.init_db()
    processor = FileProcessor()
    embedding_handler = EmbeddingHandler()
    llm = LLMHandler()

    # Tabs
    tab1, tab2 = st.tabs(["Upload Files", "API Testing"])

    # Tab 1: File Upload
    with tab1:
        st.header("Upload YAML Files")
        
        # Display existing files
        files = db.get_uploaded_files()
        st.subheader("Uploaded Files")
        for file in files:
            st.write(f"{file['filename']} - {file['upload_timestamp']}")

        # File upload
        uploaded_file = st.file_uploader("Upload YAML/YML file", type=['yaml', 'yml'])
        if uploaded_file and st.button("Process File"):
            with st.spinner("Processing file..."):
                apis, filename = processor.process_uploaded_file(uploaded_file)
                file_id = db.save_file_record(filename)
                
                for api in apis:
                    db.save_api_data(file_id, api)
                    embedding_handler.create_embedding(api)
                
                st.success("File processed successfully!")
                st.rerun()

    # Tab 2: API Testing
    with tab2:
        st.header("API Testing")
        
        # Get user query
        user_query = st.text_input("Enter your query:", key="api_test_query")
        
        if user_query:
            with st.spinner("Processing query..."):
                # Get response from LLM
                response_text = llm.get_response(user_query)
                
                if response_text:
                    try:
                        # Display the API details
                        st.subheader("API Details")
                        st.markdown(response_text)
                        
                        # Parse the response to extract API details
                        api_details = {}
                        current_key = None
                        current_value = []
                        
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                                
                            if line.startswith('-'):
                                # If we have a previous key-value pair, save it
                                if current_key and current_value:
                                    api_details[current_key] = ' '.join(current_value)
                                    current_value = []
                                
                                # Start new key-value pair
                                parts = line[2:].split(':', 1)
                                if len(parts) == 2:
                                    current_key = parts[0].strip()
                                    current_value = [parts[1].strip()]
                            elif current_key:
                                # Continue building the current value
                                current_value.append(line)
                        
                        # Save the last key-value pair
                        if current_key and current_value:
                            api_details[current_key] = ' '.join(current_value)
                        
                        # Make API call
                        if st.button("Call API"):
                            try:
                                # Prepare API data for the call
                                api_data = {
                                    'method': api_details.get('API Method', '').upper(),
                                    'full_path_with_params': api_details.get('API Full Path with parameters', ''),
                                    'request_body_dict': None
                                }
                                
                                # Parse request body if present
                                request_body_str = api_details.get('API Request Body')
                                if request_body_str and request_body_str != 'None':
                                    try:
                                        # Clean up the request body string
                                        request_body_str = request_body_str.strip()
                                        # Parse the JSON
                                        api_data['request_body_dict'] = json.loads(request_body_str)
                                    except json.JSONDecodeError as e:
                                        st.error(f"Error parsing request body: {str(e)}")
                                        st.write(f"Request body string: {request_body_str}")
                                        return
                                
                                # Call the API
                                response = call_api(api_data)
                                
                                # Display API response
                                st.subheader("API Response")
                                if 'error' in response:
                                    st.error(f"Error: {response['error']}")
                                else:
                                    # Display status code
                                    st.write(f"**Status Code:** {response['status_code']}")
                                    
                                    # Get summary from LLM
                                    with st.spinner("Summarizing response..."):
                                        summary = llm.summarize_api_response(response)
                                        st.subheader("Response Summary")
                                        st.write(summary)
                                    
                                    # Display detailed response
                                    st.subheader("Detailed Response")
                                    if response['body']:
                                        st.json(response['body'])
                                    else:
                                        st.write("No response body")
                                    
                                    # Display headers if present
                                    if response.get('headers'):
                                        st.subheader("Response Headers")
                                        st.json(response['headers'])
                                    
                            except Exception as e:
                                st.error(f"Error making API call: {str(e)}")
                    except Exception as e:
                        st.error(f"Error processing response: {str(e)}")
                        st.write("Raw response:", response_text)

def call_api(api_data):
    """Call the API with the given data."""
    try:
        # Extract API details
        method = api_data['method']
        url = api_data['full_path_with_params']
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Prepare request body for POST/PUT requests
        data = None
        if method in ['POST', 'PUT'] and 'request_body_dict' in api_data:
            data = api_data['request_body_dict']
        
        # Make the API call
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data  # Use json parameter for proper JSON serialization
        )
        
        # Return the response
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'body': response.json() if response.text else None
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'status_code': 500
        }

if __name__ == "__main__":
    main()
