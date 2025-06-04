from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.module3_gnn.gnn import load_inference_bundle, run_full_pipeline
from app.utils.ontology_loader import load_ontology_data
from app.utils.logger import logger

import os

router = APIRouter()

BUNDLE_PATH = os.path.join("fusion_gene_gnn_bundle.pth")
gnn_model, gnn_data, node_map, go_vocab, reactome_vocab, mesh_vocab = load_inference_bundle(BUNDLE_PATH)
logger.info("✅ GNN model and data loaded at startup.")

go_ontology_dict, reactome_pathway_dict = load_ontology_data()
logger.info("✅ Ontology data loaded at startup.")

class InferenceRequest(BaseModel):
    phenotype_terms: List[str]
    top_k: int = 5

@router.post("/predict")
def predict_causal_gene(request: InferenceRequest):
    """Run the full pipeline: GNN → LLM → Meta-Reasoning."""
    result = run_full_pipeline(request.phenotype_terms, request.top_k)
    logger.info(f"\n📋 Full Pipeline Result:\n{result}")
    return result