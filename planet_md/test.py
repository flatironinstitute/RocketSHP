from pathlib import Path

for i,p in enumerate(Path(__file__).resolve().parents):
    print(i, p)
