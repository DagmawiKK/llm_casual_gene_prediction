import requests

def map_ensembl_to_symbol(ensembl_id):
    try:
        url = f"https://mygene.info/v3/query?q=ensembl.protein:{ensembl_id}&species=human&fields=symbol"
        resp = requests.get(url)
        if resp.ok:
            data = resp.json().get("hits", [])
            if data and "symbol" in data[0]:
                return data[0]["symbol"]
        return ensembl_id  # Return original ID if mapping fails
    except Exception as e:
        print(f"Error mapping Ensembl ID {ensembl_id}: {e}")
        return ensembl_id

def verify_gene_with_mygene(gene_symbol):

    try:
        # Try as gene symbol first
        url = f"https://mygene.info/v3/query?q=symbol:{gene_symbol}&species=human"
        resp = requests.get(url)
        if resp.ok:
            data = resp.json().get("hits", [])
            if any(hit.get("symbol", "").upper() == gene_symbol.upper() for hit in data):
                return True

        # Try as Ensembl protein ID
        if "." in gene_symbol:
            url = f"https://mygene.info/v3/query?q=ensembl.protein:{gene_symbol}&species=human"
            resp = requests.get(url)
            if resp.ok:
                data = resp.json().get("hits", [])
                return len(data) > 0

        return False
    except Exception as e:
        print(f"Error during API verification: {e}")
        return False