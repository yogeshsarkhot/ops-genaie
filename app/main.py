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
                        for line in response_text.split('\n'):
                            if line.startswith('-'):
                                key, value = line[2:].split(':', 1)
                                api_details[key.strip()] = value.strip()
                        
                        # Make API call
                        if st.button("Call API"):
                            try:
                                # Prepare URL with parameters
                                url = api_details.get('API Full Path with parameters')
                                if not url:
                                    st.error("No URL found in response")
                                    return
                                    
                                params = {}
                                headers = {'Content-Type': 'application/json'}
                                
                                # Prepare request body
                                request_body = None
                                request_body_str = api_details.get('API Request Body')
                                if request_body_str and request_body_str != 'None':
                                    try:
                                        request_body = json.loads(request_body_str)
                                    except json.JSONDecodeError:
                                        st.error("Invalid request body format")
                                        return
                                
                                # Make the API call based on method
                                method = api_details.get('API Method', '').upper()
                                if method == 'GET':
                                    api_response = requests.get(url, params=params, headers=headers)
                                elif method == 'POST':
                                    api_response = requests.post(url, params=params, json=request_body, headers=headers)
                                elif method == 'PUT':
                                    api_response = requests.put(url, params=params, json=request_body, headers=headers)
                                elif method == 'DELETE':
                                    api_response = requests.delete(url, params=params, headers=headers)
                                else:
                                    st.error(f"Unsupported HTTP method: {method}")
                                    return
                                
                                # Display API response
                                st.subheader("API Response")
                                st.write(f"**Status Code:** {api_response.status_code}")
                                
                                try:
                                    response_json = api_response.json()
                                    st.json(response_json)
                                except ValueError:
                                    st.write(api_response.text)
                                    
                            except requests.exceptions.RequestException as e:
                                st.error(f"Error making API call: {str(e)}")
                            except Exception as e:
                                st.error(f"Unexpected error: {str(e)}")
                    except Exception as e:
                        st.error(f"Error processing response: {str(e)}")
                        st.write("Raw response:", response_text)

if __name__ == "__main__":
    main()
