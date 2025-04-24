import streamlit as st
from app.database import Database
from app.file_processor import FileProcessor
from app.llm_handler import LLMHandler
import json
import requests
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    st.set_page_config(page_title="API Analyzer", layout="wide")
    
    # Initialize components
    db = Database()
    db.init_db()
    processor = FileProcessor()
    llm = LLMHandler()
    # Tool registry for dynamic tools, stored in session state
    if 'tool_registry' not in st.session_state:
        st.session_state['tool_registry'] = {}
    tool_registry = st.session_state['tool_registry']

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
                # --- Tool creation logic: create and register tools for each API ---
                for api in apis:
                    def make_tool(api_def):
                        def tool_func(**kwargs):
                            url = api_def['full_path']
                            method = api_def['method']
                            # Prepare parameters
                            params = {k: v for k, v in kwargs.items() if k in [p['name'] for p in api_def['parameters']]}
                            data = kwargs.get('request_body', None)
                            # Substitute path parameters in the URL
                            path_param_names = re.findall(r'\{([^{}]+)\}', url)
                            for pname in path_param_names:
                                if pname in params:
                                    url = url.replace(f'{{{pname}}}', str(params[pname]))
                                    logging.info(f"Substituted path param: {pname}={params[pname]}")
                                    params.pop(pname)
                            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
                            logging.info(f"Final request: {method} {url} | params={params} | data={data}")
                            resp = requests.request(method, url, headers=headers, params=params, json=data)
                            try:
                                body = resp.json()
                            except Exception:
                                body = resp.text
                            return {"status_code": resp.status_code, "headers": dict(resp.headers), "body": body}
                        tool_func.__api_metadata__ = api_def
                        return tool_func
                    tool_name = api.get('operation_id') or f"{api['method']}_{api['name'].replace('/', '_')}"
                    param_names = [p['name'] for p in api['parameters']]
                    has_request_body = bool(api.get('request_body')) and api['request_body'] != '{}'
                    log_msg = f"Tool registered: {tool_name}\n  Method: {api['method']}\n  Path: {api['full_path']}\n  OperationId: {api.get('operation_id')}\n  Parameters: {param_names}"
                    if has_request_body:
                        log_msg += f"\n  Request Body Schema: {api['request_body']}"
                    logging.info(log_msg)
                    tool_registry[tool_name] = make_tool(api)
                st.session_state['tool_registry'] = tool_registry
                st.success("File processed and tools created successfully!")
                st.rerun()

    # Tab 2: API Testing
    with tab2:
        st.header("API Execution")
        
        # Get user query
        user_query = st.text_input("Enter your query:", key="api_test_query")
        
        if user_query:
            with st.spinner("Processing query..."):
                # Pass dynamic tools from session state to LLMHandler
                response_text = llm.get_response(user_query, tools=st.session_state['tool_registry'])
                # Defensive: Handle None response gracefully
                if response_text is None:
                    st.error("No response returned from LLM. Please check logs or try again.")
                elif response_text.startswith("Tool called:"):
                    # Parse tool name and API result
                    tool_match = re.search(r"Tool called: (.*?)\nResult: (.*)", response_text, re.DOTALL)
                    if tool_match:
                        tool_name = tool_match.group(1)
                        api_result_raw = tool_match.group(2)
                        try:
                            # Try to parse as JSON (Python dict with single quotes is not valid JSON)
                            api_result = json.loads(api_result_raw)
                        except json.JSONDecodeError:
                            try:
                                # Replace single quotes with double quotes and try again
                                api_result = json.loads(api_result_raw.replace("'", '"'))
                            except Exception:
                                # As a last resort, try eval (unsafe for untrusted input)
                                import ast
                                try:
                                    api_result = ast.literal_eval(api_result_raw)
                                except Exception:
                                    api_result = api_result_raw
                        st.subheader("API Response")
                        st.write(f"**Tool:** {tool_name}")
                        st.json(api_result)
                        # Pass the API result to LLM for user-friendly explanation
                        with st.spinner("Summarizing response..."):
                            explanation = llm.explain_api_response(api_result if isinstance(api_result, dict) else {'body': api_result})
                            st.subheader("Response Explanation")
                            st.write(explanation)
                    else:
                        st.write(response_text)
                else:
                    st.write(response_text)

    # Debug: Show tool registry in UI
    st.sidebar.subheader("Registered Tools (Debug)")
    st.sidebar.write(list(st.session_state['tool_registry'].keys()))

if __name__ == "__main__":
    main()
