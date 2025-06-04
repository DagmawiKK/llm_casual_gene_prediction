from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.sql import func
from app.database import Base

class InferenceRequest(Base):
    __tablename__ = "inference_requests"

    id = Column(Integer, primary_key=True, index=True)
    phenotype_terms = Column(JSON, nullable=False) 
    model_version = Column(String, nullable=False)  
    input_provenance = Column(JSON, nullable=False)  
    uncertainty_score = Column(Float, nullable=True)  
    predictions = Column(JSON, nullable=False)  
    created_at = Column(DateTime(timezone=True), server_default=func.now())