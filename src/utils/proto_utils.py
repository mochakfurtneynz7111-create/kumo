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
    Leiden聚类自动确定原型数量 + 保存空间信息
    """
    import scanpy as sc
    import anndata
    
    n_patches = 0
    n_total = n_proto_patches
    
    try:
        n_patches_per_batch = (n_total + len(data_loader) - 1) // len(data_loader)
    except:
        n_patches_per_batch = 1000
    
    print(f"[Leiden] Sampling maximum of {n_total} patches: {n_patches_per_batch} each from {len(data_loader)}")
    
    patches = torch.Tensor(n_total, feature_dim)
    
    # 🔥 新增：存储每个patch的空间坐标
    patch_coords = torch.Tensor(n_total, 2)  # (x, y)
    
    # === 采样patches和坐标 ===
    for batch in tqdm(data_loader):
        if n_patches >= n_total:
            continue
        
        data = batch['img']
        coords = batch.get('coords', None)  # 尝试获取坐标
        
        with torch.no_grad():
            data_reshaped = data.reshape(-1, data.shape[-1])
            np.random.shuffle(data_reshaped)
            out = data_reshaped[:n_patches_per_batch]
        
        size = out.size(0)
        if n_patches + size > n_total:
            size = n_total - n_patches
            out = out[:size]
        
        patches[n_patches: n_patches + size] = out
        
        # 🔥 新增：保存对应的坐标
        if coords is not None:
            coords_reshaped = coords.reshape(-1, 2)
            # 使用相同的shuffle和slice
            sampled_coords = coords_reshaped[:n_patches_per_batch][:size]
            patch_coords[n_patches: n_patches + size] = sampled_coords
        else:
            # 如果没有坐标，使用假坐标（顺序索引）
            fake_coords = torch.arange(n_patches, n_patches + size).unsqueeze(1).repeat(1, 2).float()
            patch_coords[n_patches: n_patches + size] = fake_coords
        
        n_patches += size
    
    print(f"\n[Leiden] Total of {n_patches} patches aggregated")
    
    s = time.time()
    
    # === Leiden聚类 ===
    print(f"\n[Leiden] Running Leiden clustering with resolution={resolution}, n_neighbors={n_neighbors}")
    
    adata = anndata.AnnData(X=patches[:n_patches].cpu().numpy())
    
    print("[Leiden] Computing PCA...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=min(50, feature_dim-1))
    
    print("[Leiden] Computing neighbors...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=50, 
                   method='umap', metric='euclidean')
    
    print("[Leiden] Running Leiden algorithm...")
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    
    leiden_labels = adata.obs['leiden'].astype(int).values
    n_proto = len(np.unique(leiden_labels))
    
    print(f"[Leiden] ✓ Automatically determined {n_proto} prototypes!")
    
    # === 计算原型中心（特征和空间） ===
    centroids = []
    proto_spatial_centers = []
    proto_spatial_spreads = []
    proto_to_patches = {}
    
    for c in range(n_proto):
        mask = leiden_labels == c
        
        # 特征中心
        cluster_patches = patches[:n_patches][mask]
        centroid = cluster_patches.mean(dim=0)
        centroids.append(centroid)
        
        # 🔥 新增：空间中心
        cluster_coords = patch_coords[:n_patches][mask]
        spatial_center = cluster_coords.mean(dim=0)  # (2,)
        proto_spatial_centers.append(spatial_center)
        
        # 🔥 新增：空间分散度
        spatial_spread = cluster_coords.std(dim=0).mean().item()
        proto_spatial_spreads.append(spatial_spread)
        
        # 记录原型包含的patches索引
        proto_to_patches[c] = torch.where(torch.from_numpy(mask))[0].numpy()
    
    centroids_matrix = torch.stack(centroids)  # (n_proto, feature_dim)
    proto_spatial_centers = torch.stack(proto_spatial_centers)  # (n_proto, 2)
    proto_spatial_spreads = torch.tensor(proto_spatial_spreads)  # (n_proto,)
    
    # === 构建特征邻居图 ===
    print("[Leiden] Computing prototype feature graph...")
    proto_distances = torch.cdist(centroids_matrix, centroids_matrix)
    
    k_neighbors = min(15, n_proto - 1)
    _, topk_indices = proto_distances.topk(k_neighbors + 1, dim=-1, largest=False)
    
    feature_adjacency = torch.zeros(n_proto, n_proto)
    for i in range(n_proto):
        neighbors = topk_indices[i, 1:]  # 跳过自己
        feature_adjacency[i, neighbors] = 1
        feature_adjacency[neighbors, i] = 1  # 对称化
    
    # 🔥 新增：构建空间邻居图
    print("[Leiden] Computing prototype spatial graph...")
    spatial_distances = torch.cdist(proto_spatial_centers, proto_spatial_centers)
    
    k_spatial = min(10, n_proto - 1)
    _, spatial_topk = spatial_distances.topk(k_spatial + 1, dim=-1, largest=False)
    
    spatial_adjacency = torch.zeros(n_proto, n_proto)
    for i in range(n_proto):
        neighbors = spatial_topk[i, 1:]
        spatial_adjacency[i, neighbors] = 1
        spatial_adjacency[neighbors, i] = 1
    
    # === 保存权重 ===
    weight = centroids_matrix.unsqueeze(0).numpy()  # [1, C, feature_dim]
    
    e = time.time()
    print(f"[Leiden] Clustering took {e-s:.2f} seconds!")
    print(f"[Leiden] Cluster sizes: {[(leiden_labels==c).sum() for c in range(n_proto)]}")
    print(f"[Leiden] Feature graph edges: {feature_adjacency.sum().item():.0f}")
    print(f"[Leiden] Spatial graph edges: {spatial_adjacency.sum().item():.0f}")
    print(f"[Leiden] Spatial spread: mean={proto_spatial_spreads.mean():.2f}, std={proto_spatial_spreads.std():.2f}")
    
    # 🔥 新增：返回完整信息字典
    extra_info = {
        'leiden_labels': leiden_labels,
        'proto_to_patches': proto_to_patches,
        'feature_adjacency': feature_adjacency.numpy(),
        'spatial_centers': proto_spatial_centers.numpy(),
        'spatial_spreads': proto_spatial_spreads.numpy(),
        'spatial_adjacency': spatial_adjacency.numpy(),
        'leiden_resolution': resolution,
        'leiden_neighbors': n_neighbors,
        'patch_coords': patch_coords[:n_patches].numpy()  # 可选：保存原始坐标
    }
    
    return n_patches, weight, n_proto, extra_info  # 返回4个值

def cluster_leiden_HPL(data_loader, feature_dim=1024, n_proto_patches=50000, 
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

