import streamlit as st
from app.database import Database
from app.file_processor import FileProcessor
from app.embeddings import EmbeddingHandler
from app.llm_handler import LLMHandler

def main():
    st.set_page_config(page_title="API Analyzer", layout="wide")
    
    # Initialize components
    db = Database()
    db.init_db()
    processor = FileProcessor()
    embedding_handler = EmbeddingHandler()
    llm = LLMHandler()

    # Tabs
    tab1, tab2 = st.tabs(["Upload Files", "Query LLM"])

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

    # Tab 2: LLM Query
    with tab2:
        st.header("Query LLM")
        
        # Query input
        query = st.text_area("Enter your query")
        if st.button("Submit Query"):
            with st.spinner("Processing query..."):
                # Get LLM response
                llm_response = llm.get_response(query)
                
                # Find relevant API
                api_match = embedding_handler.search_similar(query)
                
                # Display results
                st.write("LLM Response:", llm_response)
                if api_match:
                    st.subheader("Relevant API Found:")
                    st.json(api_match)

        # Display history
        st.subheader("Query History")
        for entry in llm.get_history():
            st.write(f"Q: {entry['query']}")
            st.write(f"A: {entry['response']}")
            st.markdown("---")

if __name__ == "__main__":
    main()
