import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from Bio import AlignIO
from Bio.Align import AlignInfo
from pathlib import Path

try:
    align_3di = sys.argv[1]
except ValueError:
    print('usage: python foldmason_to_entropy.py [foldmason 3di file]')

def pssm_to_numpy(pssm):
    npm = []
    for i, r in enumerate(pssm):
        npm.append(list(r.values()))

    return np.array(npm).T

align = AlignIO.read(align_3di, "fasta")
info = AlignInfo.SummaryInfo(align)
pssm = info.pos_specific_score_matrix()
npm = pssm_to_numpy(pssm)
pssm_dist = npm / npm.sum(0)
pssm_ent = entropy(pssm_dist)

print(pssm_to_numpy(pssm))
print(pssm_dist)
print(pssm_dist.shape)
print(pssm_dist.sum(0))
print(pssm_dist.sum(0).shape)
print(pssm_ent)
print(pssm_ent.shape)

plt.bar(np.arange(npm.shape[1]), pssm_ent)
plt.title(Path(align_3di).parent.stem)
plt.savefig(Path(align_3di).with_suffix(".entropy.png"))

np.save(Path(align_3di).with_suffix(".entropy.npy"), pssm_ent)
