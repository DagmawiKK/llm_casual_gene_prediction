import os
import torch
import logging
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel

# Setup logging (using the shared logger)
from app.utils.logger import logger

# Constants
MODEL_NAME = "dmis-lab/biobert-v1.1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# Define paths relative to the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_MERGED_MODEL_PATH = os.path.join(ROOT_DIR, "biobert_finetuned_model_merged")
OUTPUT_ADAPTER_PATH = os.path.join(ROOT_DIR, "biobert_finetuned_adapter")

def load_finetuned_model_for_inference():
    """Load fine-tuned BioBERT model (merged or adapter-based)."""
    if os.path.exists(OUTPUT_MERGED_MODEL_PATH):
        model = AutoModelForMaskedLM.from_pretrained(OUTPUT_MERGED_MODEL_PATH).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_MERGED_MODEL_PATH)
        logger.info("✅ Loaded merged fine-tuned BioBERT model.")
    elif os.path.exists(OUTPUT_ADAPTER_PATH):
        base_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = PeftModel.from_pretrained(base_model, OUTPUT_ADAPTER_PATH).to(DEVICE)
        logger.info("✅ Loaded adapter-based fine-tuned BioBERT model.")
    else:
        raise FileNotFoundError("❌ No fine-tuned model found in expected directories.")

    model.eval()
    return model, tokenizer

def compute_token_entropy(logits):
    """Compute average token entropy for uncertainty estimation."""
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy.mean().item()

def predict_causal_gene_llm(phenotype_terms, candidate_genes, llm_model, llm_tokenizer, return_confidence=False):
    """
    Use BioBERT to rerank candidate genes based on phenotype.
    If return_confidence is True, return entropy score as well.
    """
    phenotype_str = "; ".join(phenotype_terms)
    query_text = f"[CLS] Phenotype: {phenotype_str} [SEP]"

    # Encode phenotype query
    query_inputs = llm_tokenizer(query_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
    with torch.no_grad():
        query_outputs = llm_model(**query_inputs, output_hidden_states=True)
        query_embedding = query_outputs.hidden_states[-1][:, 0, :].squeeze(0)
        entropy_score = compute_token_entropy(query_outputs.logits) if return_confidence else None

    gene_embeddings = []
    for gene in candidate_genes:
        gene_text = f"[CLS] Candidate Gene: {gene} [SEP]"
        gene_inputs = llm_tokenizer(gene_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)

        with torch.no_grad():
            gene_outputs = llm_model(**gene_inputs, output_hidden_states=True)
            gene_embedding = gene_outputs.hidden_states[-1][:, 0, :].squeeze(0)
            gene_embeddings.append(gene_embedding)

    if not gene_embeddings:
        return "No candidate genes provided."

    gene_embeddings = torch.stack(gene_embeddings)
    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), gene_embeddings, dim=-1)

    best_gene_idx = torch.argmax(similarities).item()
    predicted_gene = candidate_genes[best_gene_idx]

    logger.info(f"\n🧬 LLM Inference Input: {query_text}")
    logger.info(f"\n🧬 Candidate Genes: {candidate_genes}")
    logger.info(f"\n📈 Similarity Scores: {[round(s.item(), 4) for s in similarities]}")
    logger.info(f"\n✅ Predicted Causal Gene: {predicted_gene}\n")

    if return_confidence:
        return predicted_gene, entropy_score
    return predicted_gene

# Load model once on import
finetuned_llm_model, finetuned_llm_tokenizer = load_finetuned_model_for_inference()