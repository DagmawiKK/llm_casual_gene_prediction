import os
import obonet
from app.utils.logger import logger

def load_ontology_data():

    go_obo_path = "./data/go.obo"
    reactome_path = "./data/ReactomePathways.txt"
    go_dict = {}
    reactome_dict = {}

    # Load GO terms from go.obo
    if os.path.exists(go_obo_path):
        try:
            graph = obonet.read_obo(go_obo_path)
            go_terms = {node: data.get("name", "") for node, data in graph.nodes(data=True) if node.startswith("GO:")}
            logger.info(f"✅ Loaded {len(go_terms)} GO terms from {go_obo_path}")
            logger.warning(
                "⚠️ go.obo provides ontology structure but not gene associations. "
                "To map genes to GO terms, provide a GAF file (e.g., goa_human.gaf) and extend this function."
            )
        except Exception as e:
            logger.error(f"❌ Error parsing {go_obo_path}: {e}")
    else:
        logger.warning(f"⚠️ GO ontology file not found at {go_obo_path}")

    if os.path.exists(reactome_path):
        try:
            with open(reactome_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): 
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:  
                        logger.warning(f"Skipping malformed line in ReactomePathways.txt: {line}")
                        continue
                    gene, pathway = parts[0], parts[1]
                    reactome_dict.setdefault(gene, []).append(pathway)
            logger.info(f"✅ Loaded {len(reactome_dict)} gene-pathway pairs from {reactome_path}")
        except Exception as e:
            logger.error(f"❌ Error parsing {reactome_path}: {e}")
    else:
        logger.warning(f"⚠️ Reactome pathways file not found at {reactome_path}")

    return go_dict, reactome_dict