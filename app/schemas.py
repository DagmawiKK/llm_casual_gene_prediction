from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class PhenotypeRequest(BaseModel):
    phenotype_terms: List[str]
    top_k: int = 5

class InferenceResponse(BaseModel):
    request_id: int
    candidate_genes: List[str]
    uncertainty_score: Optional[float]
    model_version: str
    input_provenance: Dict
    created_at: datetime