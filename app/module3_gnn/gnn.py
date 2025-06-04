import os
import torch
from torch_geometric.data import Data

from .model import FusionGeneGNN

from app.module1_finetune_llm.finetuned_model import (
    predict_causal_gene_llm,
    finetuned_llm_model,
    finetuned_llm_tokenizer
)
from app.module2_meta_reasoning.meta_reasoner import meta_reason_llm_output
from app.utils.ontology_loader import load_ontology_data
from app.utils.logger import logger

gnn_model = None
gnn_data = None
node_map = None
go_vocab = None
reactome_vocab = None
mesh_vocab = None
go_dict = None
reactome_dict = None


def load_inference_bundle(bundle_path: str):
    global gnn_model, gnn_data, node_map, go_vocab, reactome_vocab, mesh_vocab
    loaded_bundle = torch.load(bundle_path)

    model_params = loaded_bundle["model_params"]
    model = FusionGeneGNN(**model_params)
    model.load_state_dict(loaded_bundle["model_state_dict"])
    model.eval()

    loaded_x = loaded_bundle["feature_matrix_x"]
    loaded_edge_index = loaded_bundle["edge_index"]
    loaded_node_map = loaded_bundle["node_map"]
    loaded_go_vocab = loaded_bundle["go_vocab"]
    loaded_reactome_vocab = loaded_bundle["reactome_vocab"]
    loaded_mesh_vocab = loaded_bundle["mesh_vocab"]

    loaded_data = Data(x=loaded_x, edge_index=loaded_edge_index)

    gnn_model = model
    gnn_data = loaded_data
    node_map = loaded_node_map
    go_vocab = loaded_go_vocab
    reactome_vocab = loaded_reactome_vocab
    mesh_vocab = loaded_mesh_vocab

    logger.info("✅ GNN model and associated data loaded successfully.")
    logger.info(f"GNN data shape (x): {loaded_data.x.shape}, edge_index shape: {loaded_data.edge_index.shape}")
    return model, loaded_data, node_map, go_vocab, reactome_vocab, mesh_vocab


from app.utils.api_verifier import map_ensembl_to_symbol

def infer_causal_genes(
    model,
    data,
    node_map,
    go_vocab,
    reactome_vocab,
    mesh_vocab,
    phenotype_terms: list,
    top_k: int = 5
) -> list:
    logger.info("Starting GNN forward pass...")
    with torch.no_grad():
        embeddings = model(data)

    logger.info("Phenotype vector calculated. Shape: " + str(embeddings.size(1)))
    pheno_vec = torch.zeros(embeddings.size(1))

    for term in phenotype_terms:
        if term in go_vocab:
            pheno_vec += data.x[:, go_vocab[term]].mean(0)
        elif term in reactome_vocab:
            offset = len(go_vocab)
            pheno_vec += data.x[:, offset + reactome_vocab[term]].mean(0)
        elif term in mesh_vocab:
            offset = len(go_vocab) + len(reactome_vocab)
            pheno_vec += data.x[:, offset + mesh_vocab[term]].mean(0)

    logger.info("Calculating similarities...")
    similarities = torch.matmul(embeddings, pheno_vec)
    logger.info(f"Top {top_k} indices found: {similarities.topk(min(top_k, embeddings.size(0))).indices}")
    top_idxs = similarities.topk(min(top_k, embeddings.size(0))).indices

    reverse_map = {v: k for k, v in node_map.items()}
    candidate_genes = [reverse_map[i.item()] for i in top_idxs]

    # Map Ensembl IDs to gene symbols
    mapped_genes = [map_ensembl_to_symbol(gene) for gene in candidate_genes]
    logger.info(f"\n🧬 Top {top_k} candidate genes from GNN:\n{mapped_genes}")
    logger.warning(
        "⚠️ Some Ensembl IDs may not map to gene symbols if not found in MyGene.info."
    )
    return mapped_genes


def run_full_pipeline(
    phenotype_terms: list,
    top_k: int = 5
) -> dict:

    global go_dict, reactome_dict
    if go_dict is None or reactome_dict is None:
        go_dict, reactome_dict = load_ontology_data()

    # GNN inference
    candidate_genes = infer_causal_genes(
        gnn_model, gnn_data, node_map, go_vocab, reactome_vocab, mesh_vocab, phenotype_terms, top_k
    )

    # LLM reranking with confidence
    predicted_gene = predict_causal_gene_llm(
        phenotype_terms, candidate_genes, finetuned_llm_model, finetuned_llm_tokenizer
    )
    logger.info(f"\n🔮 LLM predicted causal gene:\n{predicted_gene}")

    # Meta-reasoning verification
    verdict = meta_reason_llm_output(
        predicted_gene,
        phenotype_terms,
        finetuned_llm_model,
        finetuned_llm_tokenizer,
        go_dict,
        reactome_dict
    )
    logger.info(f"\n🧠 Meta-Reasoning Verdict:\n{verdict}")

    return {
        "phenotype_terms": phenotype_terms,
        "gnn_candidates": candidate_genes,
        "llm_prediction": predicted_gene,
        "verification": verdict
    }

if __name__ == "__main__":
    # For local testing only
    DATA_DIR = ".."
    model_bundle_path = os.path.join(DATA_DIR, "fusion_gene_gnn_bundle.pth")

    # Load model and data
    load_inference_bundle(model_bundle_path)