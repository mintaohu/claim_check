from torch.utils.data import Dataset
from data_loader import load_evidences, load_training_data, load_claim_with_evidence


class ClaimWithEvidence(Dataset):
    def __init__(self, train_claims_path, evidences_path):
        training_data = load_training_data(train_claims_path)
        evidences = load_evidences(evidences_path)
        self.claim_with_evidence = load_claim_with_evidence(training_data, evidences)

    def __len__(self):
        return len(self.claims_and_evidence)

    def __getitem__(self, idx):
        return self.claim_with_evidence[idx]['texts'],  self.claim_with_evidence[idx]['label']