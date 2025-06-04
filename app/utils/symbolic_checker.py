def symbolic_verify_gene(predicted_gene, phenotype_terms, go_dict, reactome_dict):
    go_terms = go_dict.get(predicted_gene, [])
    reactome_terms = reactome_dict.get(predicted_gene, [])

    for term in phenotype_terms:
        if term in go_terms or term in reactome_terms:
            return True
    return False
