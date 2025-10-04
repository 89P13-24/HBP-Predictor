import requests
import csv
from io import StringIO
from Bio import SeqIO

def fetch_uniprot(query, max_sequences=2000):
    """
    Fetch protein sequences from UniProt REST API using a query.
    Returns a list of tuples: (accession_id, sequence)
    """
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "fasta",
        "query": query
    }

    print(f"Fetching from UniProt...\nQuery: {query}")
    resp = requests.get(url, params=params)

    if resp.status_code != 200:
        print(f"⚠️ Request failed: {resp.status_code} - {resp.text[:200]}")
        return []

    records = list(SeqIO.parse(StringIO(resp.text), "fasta"))
    data = []
    for record in records[:max_sequences]:
        acc = record.id.split("|")[1] if "|" in record.id else record.id
        seq = str(record.seq)
        data.append((acc, seq))

    print(f"✓ Collected {len(data)} sequences.")
    return data


# Fetch heme-binding proteins (positives)
pos_query = "reviewed:true AND keyword:heme"
positives = fetch_uniprot(pos_query, max_sequences=2000)
print(f"Collected {len(positives)} heme-binding proteins.\n")

# Fetch non-heme-binding proteins (negatives)
neg_query = "reviewed:true AND NOT keyword:heme AND length:[50 TO 500]"
negatives = fetch_uniprot(neg_query, max_sequences=2000)
print(f"Collected {len(negatives)} non-heme-binding proteins.\n")

# Save combined dataset
output_file = "heme_dataset_uniprot.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["protein_id", "sequence", "label"])
    for acc, seq in positives:
        writer.writerow([acc, seq, 1])  # label 1 = heme-binding
    for acc, seq in negatives:
        writer.writerow([acc, seq, 0])  # label 0 = non-heme-binding

print(f"✅ Dataset saved as {output_file}")
