"""
All the functions related to clustering and slide embedding construction
"""

import pdb
import os
from utils.file_utils import save_pkl, load_pkl
import numpy as np
import time
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cluster_leiden(data_loader, feature_dim=1024, n_proto_patches=50000, 
                   resolution=1.0, n_neighbors=15, use_cuda=False):
    """
    Leiden聚类自动确定原型数量
    
    参数说明:
        resolution: Leiden分辨率,控制聚类粒度
                   - 越大 → 聚类数越多 (如2.0可能产生30+个聚类)
                   - 越小 → 聚类数越少 (如0.5可能产生8-12个聚类)
                   - 推荐范围: 0.8-1.5
        n_neighbors: 邻居数,用于构建KNN图
    
    返回:
        n_patches: 实际采样的patch数量
        weight: 原型中心 [1, C, feature_dim]
        n_proto: 自动确定的原型数量
    """
    import scanpy as sc
    import anndata
    import pandas as pd
    
    n_patches = 0
    n_total = n_proto_patches  # 不再基于固定n_proto
    
    # 采样patches
    try:
        n_patches_per_batch = (n_total + len(data_loader) - 1) // len(data_loader)
    except:
        n_patches_per_batch = 1000
    
    print(f"[Leiden] Sampling maximum of {n_total} patches: {n_patches_per_batch} each from {len(data_loader)}")
    
    patches = torch.Tensor(n_total, feature_dim)
    
    for batch in tqdm(data_loader):
        if n_patches >= n_total:
            continue
        
        data = batch['img']
        with torch.no_grad():
            data_reshaped = data.reshape(-1, data.shape[-1])
            np.random.shuffle(data_reshaped)
            out = data_reshaped[:n_patches_per_batch]
        
        size = out.size(0)
        if n_patches + size > n_total:
            size = n_total - n_patches
            out = out[:size]
        patches[n_patches: n_patches + size] = out
        n_patches += size
    
    print(f"\n[Leiden] Total of {n_patches} patches aggregated")
    
    s = time.time()
    
    # === Leiden聚类核心代码 ===
    print(f"\n[Leiden] Running Leiden clustering with resolution={resolution}, n_neighbors={n_neighbors}")
    
    # 1. 创建AnnData对象
    adata = anndata.AnnData(X=patches[:n_patches].cpu().numpy())
    
    # 2. 构建邻居图(使用PCA降维以加速)
    print("[Leiden] Computing PCA...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=min(50, feature_dim-1))
    
    print("[Leiden] Computing neighbors...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=50, 
                   method='umap', metric='euclidean')
    
    # 3. Leiden聚类
    print("[Leiden] Running Leiden algorithm...")
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    
    # 4. 提取聚类标签
    leiden_labels = adata.obs['leiden'].astype(int).values
    n_proto = len(np.unique(leiden_labels))  # 自适应确定的聚类数!
    
    print(f"[Leiden] ✓ Automatically determined {n_proto} prototypes!")
    
    # 5. 计算每个聚类的中心作为原型
    centroids = []
    for c in range(n_proto):
        mask = leiden_labels == c
        cluster_patches = patches[:n_patches][mask]
        centroid = cluster_patches.mean(dim=0)
        centroids.append(centroid)
    
    weight = torch.stack(centroids).unsqueeze(0).numpy()  # [1, C, feature_dim]
    
    e = time.time()
    print(f"[Leiden] Clustering took {e-s:.2f} seconds!")
    print(f"[Leiden] Cluster sizes: {[(leiden_labels==c).sum() for c in range(n_proto)]}")
    
    return n_patches, weight, n_proto  # 注意返回3个值!


def cluster(data_loader, n_proto, n_iter, n_init=5, feature_dim=1024, 
            n_proto_patches=50000, mode='kmeans', use_cuda=False,
            leiden_resolution=1.0, leiden_neighbors=15):  # 新增参数
    """
    K-Means或Leiden clustering on embedding space
    
    mode: 'kmeans', 'faiss', 'leiden'
    """
    
    # === 新增: Leiden模式 ===
    if mode == 'leiden':
        return cluster_leiden(
            data_loader, 
            feature_dim=feature_dim,
            n_proto_patches=n_proto_patches,
            resolution=leiden_resolution,
            n_neighbors=leiden_neighbors,
            use_cuda=use_cuda
        )
    # === 原有K-means代码保持不变 ===
    
    n_patches = 0
    n_total = n_proto * n_proto_patches

    # Sample equal number of patch features from each WSI
    try:
        n_patches_per_batch = (n_total + len(data_loader) - 1) // len(data_loader)
    except:
        n_patches_per_batch = 1000

    print(f"Sampling maximum of {n_proto * n_proto_patches} patches: {n_patches_per_batch} each from {len(data_loader)}")

    patches = torch.Tensor(n_total, feature_dim)

    for batch in tqdm(data_loader):
        if n_patches >= n_total:
            continue

        data = batch['img'] # (n_batch, n_instances, instance_dim)

        with torch.no_grad():
            data_reshaped = data.reshape(-1, data.shape[-1])
            np.random.shuffle(data_reshaped)
            out = data_reshaped[:n_patches_per_batch]  # Remove batch dim

        size = out.size(0)
        if n_patches + size > n_total:
            size = n_total - n_patches
            out = out[:size]
        patches[n_patches: n_patches + size] = out
        n_patches += size

    print(f"\nTotal of {n_patches} patches aggregated")

    s = time.time()
    if mode == 'kmeans':
        print("\nUsing Kmeans for clustering...")
        print(f"\n\tNum of clusters {n_proto}, num of iter {n_iter}")
        kmeans = KMeans(n_clusters=n_proto, max_iter=n_iter)
        kmeans.fit(patches[:n_patches].cpu())
        weight = kmeans.cluster_centers_[np.newaxis, ...]

    elif mode == 'faiss':
        assert use_cuda, f"FAISS requires access to GPU. Please enable use_cuda"
        try:
            import faiss
        except ImportError:
            print("FAISS not installed. Please use KMeans option!")
            raise
        
        numOfGPUs = torch.cuda.device_count()
        print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...")
        print(f"\tNum of clusters {n_proto}, num of iter {n_iter}")

        kmeans = faiss.Kmeans(patches.shape[1], 
                              n_proto, 
                              niter=n_iter, 
                              nredo=n_init,
                              verbose=True, 
                              max_points_per_centroid=n_proto_patches,
                              gpu=numOfGPUs)
        
        kmeans.train(patches.numpy())
        weight = kmeans.centroids[np.newaxis, ...]

    else:
        raise NotImplementedError(f"Clustering not implemented for {mode}!")

    e = time.time()
    print(f"\nClustering took {e-s} seconds!")

    return n_patches, weight

def check_prototypes(n_proto, embed_dim, load_proto, proto_path):
    """
    Check validity of the prototypes
    """
    if load_proto:
        assert os.path.exists(proto_path), "{} does not exist!".format(proto_path)
        if proto_path.endswith('pkl'):
            prototypes = load_pkl(proto_path)['prototypes'].squeeze()
        elif proto_path.endswith('npy'):
            prototypes = np.load(proto_path)


        assert (n_proto == prototypes.shape[0]) and (embed_dim == prototypes.shape[1]),\
            "Prototype dimensions do not match! Params: ({}, {}) Suplied: ({}, {})".format(n_proto,
                                                                                           embed_dim,
                                                                                           prototypes.shape[0],
                                                                                           prototypes.shape[1])

