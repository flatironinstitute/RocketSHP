import os
import sys
import re
from pathlib import Path

try:
    pfile = sys.argv[1]
except ValueError:
    print('usage: python ensemble_pdb_to_dir.py [ensemble pdb file]')

basename = Path(pfile).stem

with open(pfile, 'r') as f:
    models = []
    currmodel = []

    f.readline()
    for line in f:
        if line.startswith("MODEL"):
            models.append(''.join(currmodel))
            currmodel = []
        currmodel.append(line)
    models.append(''.join(currmodels))

os.makedirs(f"ensembles/{basename}", exist_ok=True)

for i, m in enumerate(models):
    with open(f"ensembles/{basename}/{basename}_{i}.pdb", "w+") as f:
        f.write(m)
