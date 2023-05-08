import pandas as pd
from utils import filter_english_character
from ClaimWithEvidence import ClaimWithEvidence

# Convert evidences json to pandas dataframe
def load_evidences(evidences_path):
    evidences = pd.read_json(evidences_path, typ='series')
    return evidences


# Convert training claims json to pandas dataframe
def load_claims(claims_path):
    claims = pd.read_json(claims_path).T
    return claims


def load_claim_with_evidence(training_data, evidences):
    claim_with_evidence = []
    for i in range(len(training_data)):
        claim = training_data.iloc[i]
        for evidence_idx_str in claim['evidences']:
            evidence_idx = int(evidence_idx_str.replace('evidence-', ''))
            evidence = filter_english_character(evidences.iloc[evidence_idx])
            claim_with_evidence.append(ClaimWithEvidence([claim['claim_text'], evidence],1.0))

    return claim_with_evidence


