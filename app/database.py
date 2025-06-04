from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

from urllib.parse import quote_plus

POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
if not POSTGRES_PASSWORD:
    raise ValueError("POSTGRES_PASSWORD environment variable not set")

encoded_password = quote_plus(POSTGRES_PASSWORD)
print(f"Loaded password: {POSTGRES_PASSWORD} -> Encoded: {encoded_password}")

DATABASE_URL = f"postgresql://postgres:{encoded_password}@localhost:5432/causal_gene_db"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()