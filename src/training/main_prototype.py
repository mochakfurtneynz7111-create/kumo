"""
This will perform K-means or Leiden clustering on the training data
Good reference for clustering
https://github.com/facebookresearch/faiss/wiki/FAQ#questions-about-training
"""
from __future__ import print_function
import argparse
import torch
from torch.utils.data import DataLoader
from wsi_datasets import WSIProtoDataset
from utils.utils import seed_torch, read_splits
from utils.file_utils import save_pkl
from utils.proto_utils import cluster
import os
from os.path import join as j_

def build_datasets(csv_splits, batch_size=1, num_workers=2, train_kwargs={}):
    dataset_splits = {}
    for k in csv_splits.keys(): # ['train']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy()
        dataset = WSIProtoDataset(df, **dataset_kwargs)
        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
    return dataset_splits

def main(args):
    
    train_kwargs = dict(data_source=args.data_source)
       
    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('\nsuccessfully read splits for: ', list(csv_splits.keys()))
    dataset_splits = build_datasets(csv_splits,
                                    batch_size=1,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs)
    print('\nInit Datasets...', end=' ')
    os.makedirs(j_(args.split_dir, 'prototypes'), exist_ok=True)
    loader_train = dataset_splits['train']
    
    # ========== 聚类: 支持K-means和Leiden ==========
    if args.mode == 'leiden':
        # Leiden模式: 返回3个值 (n_patches, weights, n_proto_actual)
        print(f"\n{'='*60}")
        print(f"Using Leiden clustering (HPL method)")
        print(f"Resolution: {args.leiden_resolution}")
        print(f"Neighbors: {args.leiden_neighbors}")
        print(f"{'='*60}\n")
        
        _, weights, n_proto_actual = cluster(
            loader_train,
            n_proto=None,  # Leiden不需要预设数量
            n_iter=args.n_iter,
            n_init=args.n_init,
            feature_dim=args.in_dim,
            mode='leiden',
            n_proto_patches=args.n_proto_patches,
            use_cuda=True if torch.cuda.is_available() else False,
            leiden_resolution=args.leiden_resolution,
            leiden_neighbors=args.leiden_neighbors
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Leiden automatically determined {n_proto_actual} prototypes!")
        print(f"  (ignoring --n_proto={args.n_proto})")
        print(f"{'='*60}\n")
        
        actual_n_proto = n_proto_actual
        
    else:
        # K-means/FAISS模式: 返回2个值 (n_patches, weights)
        print(f"\n{'='*60}")
        print(f"Using {args.mode.upper()} clustering")
        print(f"Number of prototypes: {args.n_proto} (fixed)")
        print(f"{'='*60}\n")
        
        _, weights = cluster(
            loader_train,
            n_proto=args.n_proto,
            n_iter=args.n_iter,
            n_init=args.n_init,
            feature_dim=args.in_dim,
            mode=args.mode,
            n_proto_patches=args.n_proto_patches,
            use_cuda=True if torch.cuda.is_available() else False
        )
        
        actual_n_proto = args.n_proto
    
    # ========== 保存原型 ==========
    # 生成文件名
    mode_str = f"{args.mode}_res{args.leiden_resolution:.1f}" if args.mode == 'leiden' else args.mode
    
    save_fpath = j_(args.split_dir,
                    'prototypes',
                    f"prototypes_c{actual_n_proto}_{args.data_source[0].split('/')[-2]}_{mode_str}_num_{args.n_proto_patches:.1e}.pkl")
    
    # 保存
    save_pkl(save_fpath, {
        'prototypes': weights,
        'n_proto': actual_n_proto,
        'mode': args.mode,
        'resolution': args.leiden_resolution if args.mode == 'leiden' else None
    })
    
    print(f"\n{'='*60}")
    print(f"✓ Prototypes saved to:")
    print(f"  {save_fpath}")
    print(f"  - Number of prototypes: {actual_n_proto}")
    print(f"  - Prototype shape: {weights.shape}")
    print(f"  - Clustering mode: {args.mode}")
    print(f"{'='*60}\n")

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')

# model / loss fn args ###
parser.add_argument('--n_proto', type=int, default=16,
                    help='Number of prototypes (only used for kmeans/faiss, ignored for leiden)')
parser.add_argument('--n_proto_patches', type=int, default=10000,
                    help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
parser.add_argument('--n_init', type=int, default=5,
                    help='Number of different KMeans initialization (for FAISS)')
parser.add_argument('--n_iter', type=int, default=50,
                    help='Number of iterations for Kmeans clustering')
parser.add_argument('--in_dim', type=int)
parser.add_argument('--mode', type=str, 
                    choices=['kmeans', 'faiss', 'leiden'],  # 添加leiden选项
                    default='kmeans')

# Leiden clustering parameters (only used when mode='leiden')
parser.add_argument('--leiden_resolution', type=float, default=1.0,
                    help='Leiden clustering resolution parameter (0.5-2.0). '
                         'Higher values → more clusters. Only used when mode=leiden')
parser.add_argument('--leiden_neighbors', type=int, default=15,
                    help='Number of neighbors for Leiden graph construction. '
                         'Only used when mode=leiden')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

if __name__ == "__main__":
    args.split_dir = j_('splits', args.split_dir)
    args.split_name = os.path.basename(args.split_dir)
    print('split_dir: ', args.split_dir)
    args.data_source = [src for src in args.data_source.split(',')]
    results = main(args)
