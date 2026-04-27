import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from tqdm import tqdm
import numpy as np

# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
df = pd.read_csv("simulant_with_TIC_parent.csv")  # 여기엔 SMILES, Parent_TIC_Index 포함

# 유효 SMILES만 필터
df = df.dropna(subset=["SMILES", "Parent_TIC_Index"]).reset_index(drop=True)

# ---------------------------
# 2. Fingerprint 계산
# ---------------------------
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

print("Calculating fingerprints...")
fps = [smiles_to_fp(smi) for smi in tqdm(df["SMILES"])]
valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
df = df.iloc[valid_idx].reset_index(drop=True)
fps = [fps[i] for i in valid_idx]
fps_array = np.array([list(fp) for fp in fps])

# ---------------------------
# 3. t-SNE
# ---------------------------
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(fps_array)
df["tSNE-1"] = tsne_result[:, 0]
df["tSNE-2"] = tsne_result[:, 1]

# ---------------------------
# 4. 시각화
# ---------------------------
# 클러스터 수만큼 colormap 설정
unique_clusters = sorted(df["Parent_TIC_Index"].unique())
n_clusters = len(unique_clusters)
cmap = get_cmap("tab20", n_clusters)  # 최대 20개씩 나눠 반복됨

# Parent_TIC_Index를 색상 매핑용 숫자로 인코딩
cluster_to_color_idx = {tic: idx for idx, tic in enumerate(unique_clusters)}
df["ColorIndex"] = df["Parent_TIC_Index"].map(cluster_to_color_idx)

# 그리기
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df["tSNE-1"], df["tSNE-2"],
                      c=df["ColorIndex"], cmap=cmap,
                      alpha=0.7, s=12)

plt.title("t-SNE Colored by Most Similar TIC Parent Cluster")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label="Parent TIC Cluster Index")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
plt.savefig("tsne_clustered.png", dpi=300)
