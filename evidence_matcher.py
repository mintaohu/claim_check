from sentence_transformers import SentenceTransformer, models, losses
import torch
from torch.utils.data import DataLoader
from data_loader import load_claims, load_evidences, load_claim_with_evidence


train_claims_path = 'data/train-claims.json'
evidences_path = 'data/evidence.json'
save_path = 'weights/evidence_finder.pth'

backbone = models.Transformer('albert-large-v2', max_seq_length=256)
pooling_model = models.Pooling(backbone.get_word_embedding_dimension())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evidence_matcher = SentenceTransformer(modules=[backbone, pooling_model], device=device)
training_claims = load_claims(train_claims_path)
evidences = load_evidences(evidences_path)
training_data = load_claim_with_evidence(training_claims, evidences)
train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(evidence_matcher)

evidence_matcher.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)
evidence_matcher.save(save_path)