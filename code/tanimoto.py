import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Load data
tic_df = pd.read_csv("TIC.CSV", encoding="ISO-8859-1", header=None, names=["SMILES"])
sim_df = pd.read_csv("simulant.CSV", encoding="ISO-8859-1", header=None, names=["SMILES"])

# 2. Convert SMILES to fingerprints
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

print("Calculating fingerprints...")
tic_fps = [smiles_to_fp(smi) for smi in tqdm(tic_df["SMILES"])]
sim_fps = [smiles_to_fp(smi) for smi in tqdm(sim_df["SMILES"])]

# Filter out invalid entries
tic_fps = [fp for fp in tic_fps if fp is not None]
sim_fps = [fp for fp in sim_fps if fp is not None]

# 3. Compute maximum Tanimoto similarity to any TIC
print("Computing Tanimoto similarities...")
max_similarities = []
for sim_fp in tqdm(sim_fps):
    sims = DataStructs.BulkTanimotoSimilarity(sim_fp, tic_fps)
    max_similarities.append(max(sims))

# 4. Plot histogram
plt.figure(figsize=(10, 8))
plt.hist(max_similarities, bins=30, color="#d62728", edgecolor="black", alpha=0.8)
plt.title("Maximum Tanimoto Similarity to TICs (n = {})".format(len(sim_fps)))
plt.xlabel("Max Tanimoto Similarity")
plt.ylabel("Number of Simulants")
plt.grid(False)
plt.tight_layout()
plt.show()
plt.savefig("tanimoto.png", dpi=300)
