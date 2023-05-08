import pandas as pd
import string
# printable = set(string.printable)

evidences = pd.read_json('data/evidence.json', typ='series')
# training_data = pd.read_json('data/train-claims.json').T
evidence = evidences.iloc[9]
print(evidence)
# a = ''.join(filter(lambda x: x in printable, evidence))
# claim = training_data.iloc[0]
#
# for evidence_idx_str in claim['evidences']:
#     evidence_idx = int(evidence_idx_str.replace('evidence-',''))
