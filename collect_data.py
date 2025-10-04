import requests
import csv
from io import StringIO
from Bio import SeqIO
import argparse

def fetch_uniprot(query, max_sequences=2000, max_length=None):
    """
    Fetch protein sequences from UniProt REST API using a query.
    
    Args:
        query (str): UniProt query string
        max_sequences (int): Maximum number of sequences to fetch
        max_length (int, optional): Maximum allowed sequence length; sequences longer than this are skipped
    
    Returns:
        List of tuples: (accession_id, sequence)
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

    for record in records:
        if len(data) >= max_sequences:
            break
        
        acc = record.id.split("|")[1] if "|" in record.id else record.id
        seq = str(record.seq)

        if max_length is not None and len(seq) > max_length:
            continue  # skip sequences longer than max_length
        
        data.append((acc, seq))

    print(f"✓ Collected {len(data)} sequences.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Fetch heme-binding and non-heme proteins from UniProt")
    parser.add_argument("--max_sequences", type=int, default=2000, help="Max sequences per query (default: 2000)")
    parser.add_argument("--max_length", type=int, default=2000, help="Max sequence length (default: 500)")
    parser.add_argument("--output", type=str, default="heme_dataset_uniprot.csv", help="Output CSV file")
    args = parser.parse_args()

    max_sequences = args.max_sequences
    max_length = args.max_length
    output_file = args.output

    # Fetch heme-binding proteins (positives)
    pos_query = "reviewed:true AND keyword:heme"
    positives = fetch_uniprot(pos_query, max_sequences=max_sequences, max_length=max_length)
    print(f"Collected {len(positives)} heme-binding proteins.\n")

    # Fetch non-heme-binding proteins (negatives)
    neg_query = "reviewed:true AND NOT keyword:heme"
    negatives = fetch_uniprot(neg_query, max_sequences=max_sequences, max_length=max_length)
    print(f"Collected {len(negatives)} non-heme-binding proteins.\n")

    # Save combined dataset
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_id", "sequence", "label"])
        for acc, seq in positives:
            writer.writerow([acc, seq, 1])
        for acc, seq in negatives:
            writer.writerow([acc, seq, 0])

    print(f"✅ Dataset saved as {output_file}")


if __name__ == "__main__":
    main()
