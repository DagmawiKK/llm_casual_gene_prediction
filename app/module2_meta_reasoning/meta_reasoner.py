from app.utils.entropy_utils import compute_token_entropy
from app.utils.symbolic_checker import symbolic_verify_gene
from app.utils.api_verifier import verify_gene_with_mygene
from app.utils.logger import logger

import torch

def meta_reason_llm_output(
    predicted_gene: str,
    phenotype_terms: list,
    llm_model,
    llm_tokenizer,
    go_dict: dict,
    reactome_dict: dict
) -> dict:


    # Compute Entropy
    input_text = f"[CLS] Phenotype: {'; '.join(phenotype_terms)} Candidate Gene: {predicted_gene} [SEP]"
    inputs = llm_tokenizer(input_text, return_tensors="pt", truncation=True).to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model(**inputs)  
        entropy_score = compute_token_entropy(outputs.logits)

    # Symbolic Verification
    symbolic_valid = symbolic_verify_gene(predicted_gene, phenotype_terms, go_dict, reactome_dict)

    # External API Verification
    api_valid = verify_gene_with_mygene(predicted_gene)

    # Final Verdict
    hallucinated = not (symbolic_valid or api_valid)

    result = {
        "predicted_gene": predicted_gene,
        "entropy_score": round(entropy_score, 4),
        "symbolic_grounding": symbolic_valid,
        "api_verified": api_valid,
        "hallucination": hallucinated
    }

    logger.info("\n🧠 Meta-Reasoning Result:\n" + "\n".join(f"{k}: {v}" for k, v in result.items()))

    return result