import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json

import yaml  # Add this import

class Database:
    def __init__(self):
        self.conn = psycopg2.connect(
            database="api_analyzer",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        
    def init_db(self):
        with self.conn.cursor() as cur:
            try:
                cur.execute("""
                    DO $$ 
                    BEGIN
                        -- Create uploaded_files table if it doesn't exist
                        IF NOT EXISTS (SELECT FROM pg_tables WHERE tablename = 'uploaded_files') THEN
                            CREATE SEQUENCE IF NOT EXISTS uploaded_files_id_seq;
                            CREATE TABLE uploaded_files (
                                id INTEGER DEFAULT nextval('uploaded_files_id_seq') PRIMARY KEY,
                                filename VARCHAR(255),
                                upload_timestamp TIMESTAMP
                            );
                        END IF;

                        -- Create apis table if it doesn't exist
                        IF NOT EXISTS (SELECT FROM pg_tables WHERE tablename = 'apis') THEN
                            CREATE SEQUENCE IF NOT EXISTS apis_id_seq;
                            CREATE TABLE apis (
                                id INTEGER DEFAULT nextval('apis_id_seq') PRIMARY KEY,
                                file_id INTEGER REFERENCES uploaded_files(id),
                                api_name VARCHAR(255),
                                method VARCHAR(50),
                                summary TEXT,
                                description TEXT,
                                parameters JSONB,
                                request_body TEXT,
                                response_schemas JSONB,
                                base_url TEXT,
                                full_path TEXT
                            );
                        END IF;

                        -- Create query_history table if it doesn't exist
                        IF NOT EXISTS (SELECT FROM pg_tables WHERE tablename = 'query_history') THEN
                            CREATE SEQUENCE IF NOT EXISTS query_history_id_seq;
                            CREATE TABLE query_history (
                                id INTEGER DEFAULT nextval('query_history_id_seq') PRIMARY KEY,
                                query TEXT,
                                response TEXT,
                                timestamp TIMESTAMP
                            );
                        END IF;
                    END $$;
                """)
                self.conn.commit()
            except Exception as e:
                # If there's an error, rollback the transaction
                self.conn.rollback()
                # Log the error but don't raise it
                print(f"Database initialization warning: {str(e)}")

    def save_file_record(self, filename):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO uploaded_files (filename, upload_timestamp) VALUES (%s, %s) RETURNING id",
                (filename, datetime.now())
            )
            file_id = cur.fetchone()['id']
            self.conn.commit()
            return file_id

    def save_api_data(self, file_id, api_data):
        with self.conn.cursor() as cur:
            # Convert all dictionary fields to JSON strings
            parameters_json = json.dumps(api_data['parameters']) if api_data['parameters'] else '[]'
            response_schemas_json = json.dumps(api_data['response_schemas']) if api_data['response_schemas'] else '{}'
            tags_json = json.dumps(api_data['tags']) if api_data['tags'] else '[]'
            
            # Ensure request_body is a string (YAML or empty string)
            request_body = api_data['request_body'] if isinstance(api_data['request_body'], str) else ''
            
            cur.execute(
                """INSERT INTO apis (file_id, api_name, method, summary, description, parameters, request_body, response_schemas, base_url, full_path)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (file_id, 
                 api_data['name'], 
                 api_data['method'], 
                 api_data['summary'],
                 api_data['description'], 
                 parameters_json,
                 request_body,  # Already a YAML string
                 response_schemas_json,
                 api_data.get('base_url', ''), 
                 api_data.get('full_path', ''))
            )
            self.conn.commit()

    def get_uploaded_files(self):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM uploaded_files ORDER BY upload_timestamp DESC")
            return cur.fetchall()

    def save_query_history(self, query, response):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO query_history (query, response, timestamp) VALUES (%s, %s, %s)",
                (query, response, datetime.now())
            )
            self.conn.commit()

    def get_query_history(self):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM query_history ORDER BY timestamp DESC")
            return cur.fetchall()