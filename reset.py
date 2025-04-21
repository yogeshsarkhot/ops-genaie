import os
import shutil
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reset.log')
    ]
)

logger = logging.getLogger(__name__)

def reset_uploads():
    """Delete the uploads folder completely."""
    logger.info("Resetting uploads folder...")
    
    uploads_folder_path = os.path.join("uploads")
    
    if os.path.exists(uploads_folder_path):
        try:
            shutil.rmtree(uploads_folder_path)
            logger.info(f"Successfully deleted uploads folder at {uploads_folder_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete uploads folder: {e}")
            return False
    else:
        logger.info(f"Uploads folder not found at {uploads_folder_path}, nothing to delete")
        return True

def reset_postgres_db():
    """Reset the PostgreSQL database."""
    logger.info("Resetting PostgreSQL database...")
    
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            dbname="postgres",  # Connect to default DB first
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Drop and recreate the database
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname='api_analyzer'")
            exists = cur.fetchone()
            
            if exists:
                # Terminate existing connections
                cur.execute("""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = 'api_analyzer'
                    AND pid <> pg_backend_pid()
                """)
                
                # Drop the database
                cur.execute("DROP DATABASE api_analyzer")
                logger.info("Dropped api_analyzer database")
            
            # Create the database
            cur.execute("CREATE DATABASE api_analyzer")
            logger.info("Created fresh api_analyzer database")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to reset PostgreSQL database: {e}")
        return False

if __name__ == "__main__":
    print("Resetting database, prior uploads and starting fresh...")
    
    # Reset PostgreSQL database
    postgres_reset = reset_postgres_db()
    
    # Reset uploads folder  
    uploads_reset = reset_uploads()

    if postgres_reset and uploads_reset:
        print("Database reset successfully!")
    else:
        print("Failed to reset database. Check reset.log for details.") 