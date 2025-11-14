import os, glob, pickle, numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

feature_root = "/mnt/32TB/prh/proj/wsi_data/TCGA_BLCA"  # 修改数据集名称
pt_dir = os.path.join(feature_root, "pt_files")

seed = 42
n_clusters = 100          # 建议100或200，更贴近作者实践
use_pca = True
pca_dim = 128

rng = np.random.RandomState(seed)

fname2ids = {}
pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
for pt_path in pt_files:
    slide_id = os.path.splitext(os.path.basename(pt_path))[0]  # 去掉后缀，训练更容易匹配
    feats = torch.load(pt_path, map_location="cpu")
    if isinstance(feats, dict):
        # 兼容 {'feats': tensor} / {'features': tensor} 结构
        feats = feats.get("feats", feats.get("features", None))
        if feats is None:
            # 退而求其次：取第一个二维tensor
            for v in list(feats.values()):
                if hasattr(v, "shape") and len(v.shape) == 2:
                    feats = v; break
    feats = feats.detach().cpu().float().numpy()  # [num_patches, D]
    N, D = feats.shape

    # 可选 PCA 加速
    X = feats
    if use_pca:
        max_comp = min(N, D) - 1   # PCA 要求 n_components <= min(n_samples, n_features)
        if max_comp >= 2:
            n_comp = min(pca_dim, max_comp)
            print(f"  PCA: {D} -> {n_comp} (N={N})")
            pca = PCA(n_components=n_comp, random_state=seed)
            X = pca.fit_transform(feats).astype(np.float32)
        else:
            # 样本太少，直接跳过 PCA
            print(f"  Skip PCA for {slide_id}: N={N}, D={D}")
            X = feats.astype(np.float32, copy=False)
    else:
        X = feats.astype(np.float32, copy=False)

    # 边界：patch 少于簇数
    k = min(n_clusters, N)
    if k <= 1:
        labels = np.zeros((N,), dtype=np.int32)
    else:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=4096,
            reassignment_ratio=0.01,
            n_init="auto"  # 若遇到旧版sklearn报错，可改成 int(10)
        )
        labels = km.fit_predict(X).astype(np.int32)

    fname2ids[slide_id] = labels  # 用 slide_id 作为 key（无后缀）

out_pkl = os.path.join(feature_root, "fast_cluster_ids.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump(fname2ids, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved:", out_pkl, "slides:", len(fname2ids))
