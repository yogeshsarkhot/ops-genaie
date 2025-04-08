import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json  # Add this import

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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255),
                    upload_timestamp TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS apis (
                    id SERIAL PRIMARY KEY,
                    file_id INTEGER REFERENCES uploaded_files(id),
                    api_name VARCHAR(255),
                    method VARCHAR(50),
                    summary TEXT,
                    description TEXT,
                    parameters JSONB,
                    request_body JSONB,
                    response_schemas JSONB,
                    base_url TEXT,
                    full_path TEXT
                );

                CREATE TABLE IF NOT EXISTS query_history (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    response TEXT,
                    timestamp TIMESTAMP
                );
            """)
            self.conn.commit()

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
            cur.execute(
                """INSERT INTO apis (file_id, api_name, method, summary, description, parameters, request_body, response_schemas, base_url, full_path)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (file_id, api_data['name'], api_data['method'], api_data['summary'],
                 api_data['description'], json.dumps(api_data['parameters']),
                 json.dumps(api_data['request_body']), json.dumps(api_data['response_schemas']),
                 api_data.get('base_url', ''), api_data.get('full_path', ''))
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