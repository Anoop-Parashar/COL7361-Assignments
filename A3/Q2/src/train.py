"""
train.py  –  COL761 Assignment 3 training script

Usage
-----
python train.py --dataset A|B|C --task node|link \
    --data_dir /absolute/path/to/datasets \
    --model_dir /path/to/save/models \
    --kerberos YOUR_KERBEROS
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import dropout_edge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from load_dataset import load_dataset
from models import (
    GATModelPaper,
    TAGModelPaper,
    ARMAModelPaper,
    EnsembleModel,
    margin_ranking_loss,
    bce_loss,
    SIEGLinkPredictor,
)
from predict_wrapper_B import ChunkedNodeWrapper


# Shared utils


def expand_masks(data):
    N = data.num_nodes
    y_full = torch.full((N,), -1, dtype=torch.long)
    y_full[data.labeled_nodes] = data.y
    data.y = y_full
    train_full = torch.zeros(N, dtype=torch.bool)
    val_full = torch.zeros(N, dtype=torch.bool)
    train_full[data.labeled_nodes] = data.train_mask
    val_full[data.labeled_nodes] = data.val_mask
    data.train_mask = train_full
    data.val_mask = val_full
    return data


@torch.no_grad()
def eval_acc(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    return int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())


def hits_at_k(pos_scores, neg_scores, k=50):
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


# Dataset A  –  ensemble of GAT + TAGCN + ARMA (paper-exact hyperparameters)


TIME_LIMIT_A = 3600
PATIENCE_A = 100

# Paper-exact configs: (model_class, model_kwargs, lr, wd)
CONFIGS_A = [
    # GAT — Velickovic et al. 2018
    # 2-layer, 8 heads x 8 features, ELU, dropout=0.6, wd=5e-4
    (GATModelPaper, dict(dropout=0.6), 0.005, 5e-4),
    # TAGCN — Du et al. 2019
    # 2-layer, 16 hidden, K=3, dropout=0.5, lr=0.01
    (TAGModelPaper, dict(dropout=0.5, K=3), 0.01, 5e-4),
    # ARMA — Bianchi et al. 2021, Table 6 Cora settings
    # hidden=16, K=2 stacks, T=1 depth, dropout=0.75, wd=5e-4
    (
        ARMAModelPaper,
        dict(hidden=16, dropout=0.75, num_stacks=2, num_layers=1),
        0.01,
        5e-4,
    ),
]


def train_one_A(model_cls, mkw, lr, wd, data, device, num_classes):
    model = model_cls(data.x.size(1), num_classes, **mkw).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=50, min_lr=1e-5
    )

    best_val, patience_count, best_state = 0.0, 0, None

    for epoch in range(1, 5001):
        model.train()
        optimizer.zero_grad()
        F.cross_entropy(
            model(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        ).backward()
        optimizer.step()

        val_acc = eval_acc(model, data, data.val_mask)
        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val, patience_count = val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        if patience_count >= PATIENCE_A:
            break

    model.load_state_dict(best_state)
    return model, best_val


def train_A(dataset, model_save_path, device):
    print("\n── Training Dataset A (GAT + TAGCN + ARMA ensemble) ──")
    data = expand_masks(dataset[0]).to(device)
    num_classes = dataset.num_classes
    t0 = time.time()
    trained = []  # (model, val_acc)

    for i, (model_cls, mkw, lr, wd) in enumerate(CONFIGS_A):
        if time.time() - t0 > TIME_LIMIT_A - 120:
            print("  Time limit reached.")
            break
        name = model_cls.__name__
        print(f"  [{i + 1}/{len(CONFIGS_A)}] {name} ...", end="", flush=True)
        model, val_acc = train_one_A(model_cls, mkw, lr, wd, data, device, num_classes)
        print(f"  val={val_acc:.4f}")
        trained.append((model, val_acc))

    # best single model
    best_model, best_val = max(trained, key=lambda x: x[1])

    # only ensemble models within 1% of best
    top_models = [m for m, v in trained if v >= best_val - 0.01]

    if len(top_models) > 1:
        final = EnsembleModel(top_models)
        print(
            f"\nEnsembling {len(top_models)} models (all within 1% of best {best_val:.4f})"
        )
    else:
        final = best_model
        print(f"\nUsing best single model (val={best_val:.4f})")

    torch.save(final, model_save_path)
    print(f"  Saved -> {model_save_path}")


# Dataset B  –  GraphSAGE mini-batch


TIME_LIMIT_B = 2 * 3600


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()


def train_B(dataset, model_save_path, device):
    print("\n── Training Dataset B (GAAP, mini-batch) ──")
    from models import GAAPModelB
    import numpy as np

    data = dataset[0]
    train_node_idx = data.labeled_nodes[data.train_mask]
    val_node_idx = data.labeled_nodes[data.val_mask]

    labels = data.y.cpu().numpy().astype(np.int64)
    class_counts = np.bincount(labels)
    class_weights = torch.FloatTensor([1.0 / count for count in class_counts]).to(
        device
    )
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    print(
        f"  Class 0: {class_counts[0]}, Class 1: {class_counts[1]}, Ratio: {class_counts[1] / class_counts[0]:.4f}"
    )

    train_loader = NeighborLoader(
        data,
        num_neighbors=[3, 2],  # fewer neighbors = less mmap I/O per batch
        batch_size=2048,  # larger = fewer batches per epoch
        input_nodes=train_node_idx,
        shuffle=True,
        num_workers=0,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=[3, 2],
        batch_size=4096,
        input_nodes=val_node_idx,
        shuffle=False,
        num_workers=0,
    )

    model = GAAPModelB(data.x.shape[1], hidden=128, dropout=0.3, num_bins=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    global_cache = None
    cache_size = 10000

    best_auc, best_state = 0.0, None
    t0 = time.time()
    epoch = 0
    patience_count = 0
    max_patience = 10

    while True:
        epoch += 1
        if time.time() - t0 > TIME_LIMIT_B - 120:
            print(f"  Time budget reached at epoch {epoch}.")
            break

        # TRAINING
        model.train()
        total_loss = 0
        train_embeddings = []

        for batch in train_loader:
            batch = batch.to(device)
            seed = batch.batch_size
            optimizer.zero_grad()

            out, hidden = model(
                batch.x, batch.edge_index, global_cache, return_hidden=True
            )
            out = out[:seed]
            hidden = hidden[:seed]

            loss = focal_loss(out, batch.y[:seed].long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                probs = torch.softmax(out.detach(), dim=1)[:, 1]
                train_embeddings.append((hidden.detach().cpu(), probs.cpu()))

        # Build cache from train fraud nodes
        if len(train_embeddings) > 0:
            all_h = torch.cat([h for h, _ in train_embeddings])
            all_p = torch.cat([p for _, p in train_embeddings])
            top_k = torch.argsort(all_p, descending=True)[:cache_size]
            global_cache = all_h[top_k].to(device)

        # VALIDATION
        model.eval()
        all_scores, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                seed = batch.batch_size

                out, hidden = model(
                    batch.x, batch.edge_index, global_cache, return_hidden=True
                )
                out = out[:seed]

                probs = torch.softmax(out, dim=1)[:, 1]
                all_scores.append(probs.cpu())
                all_labels.append(batch.y[:seed].cpu())

        auc = roc_auc_score(
            torch.cat(all_labels).numpy(), torch.cat(all_scores).numpy()
        )
        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if patience_count >= max_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d} | Loss {total_loss / len(train_loader):.4f} | "
            f"AUC {auc:.4f} | Best {best_auc:.4f} | "
            f"Cache {'Yes' if global_cache is not None else 'No'} | "
            f"{elapsed / 60:.1f}min"
        )

    model.load_state_dict(best_state)
    model.eval()

    wrapped = ChunkedNodeWrapper(model.cpu(), num_nodes=data.num_nodes)
    torch.save(wrapped, model_save_path)
    print(f"\nSaved model -> {model_save_path}  (Best AUC: {best_auc:.4f})")


# Dataset C  –  link prediction


TIME_LIMIT_C = 2 * 3600


# SIEGLinkPredictor .
# https://github.com/anonymous20221001/SIEG_OGB/blob/master/OGB_VESSEL_SIEG.pdf


def train_C(dataset, model_save_path, device):
    """
    SIEG for Dataset C — bug-fixed, LayerNorm, tuned dropout.

    USE_MARGIN_LOSS = False  ->  BCEWithLogitsLoss  (paper default, start here)
    USE_MARGIN_LOSS = True   ->  Max-margin ranking loss (try if BCE plateaus)
    """
    import time

    # Toggle here
    USE_MARGIN_LOSS = True
    MARGIN = 3.0  # changed from 1.0
    # ------------------

    print("\n Training Dataset C (SIEG, bug-fixed + LayerNorm)")
    print(f"   Loss: {'MarginRanking' if USE_MARGIN_LOSS else 'BCE'}")

    x = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    train_pos = dataset.train_pos.to(device)
    train_neg = dataset.train_neg.to(device)
    valid_pos = dataset.valid_pos.to(device)
    valid_neg = dataset.valid_neg.to(device)

    model = SIEGLinkPredictor(
        in_channels=x.shape[1],
        hidden=256,
        out=128,
        dropout=0.3,  # changed from 0.5
        max_spd=10,  # changed from 5
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )  # changed from 1e-5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=30, min_lr=1e-6
    )

    TIME_LIMIT_C = 2 * 3600
    PATIENCE = 80

    def hits_at_k(pos_scores, neg_scores, k=50):
        n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
        return (n_neg_higher < k).float().mean().item()

    # For margin loss we need matched pairs
    n_pairs = min(train_pos.shape[0], train_neg.shape[0])

    best_hits, best_state = 0.0, None
    patience_count = 0
    t0 = time.time()

    for epoch in range(1, 2001):
        if time.time() - t0 > TIME_LIMIT_C - 120:
            print(f"  Time budget reached at epoch {epoch}.")
            break

        # Training
        model.train()
        optimizer.zero_grad()

        pos_scores = model(x, edge_index, train_pos)
        neg_scores = model(x, edge_index, train_neg)

        if USE_MARGIN_LOSS:
            loss = margin_ranking_loss(
                pos_scores[:n_pairs], neg_scores[:n_pairs], margin=MARGIN
            )
        else:
            loss = bce_loss(pos_scores, neg_scores, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                vp = model(x, edge_index, valid_pos)
                V, K, _ = valid_neg.shape
                CHUNK = 10  # changed from 50
                vn_chunks = []
                for start in range(0, V, CHUNK):
                    end = min(start + CHUNK, V)
                    C = end - start
                    vn_chunk = model(
                        x, edge_index, valid_neg[start:end].view(C * K, 2)
                    ).view(C, K)
                    vn_chunks.append(vn_chunk)
                vn = torch.cat(vn_chunks, dim=0)  # [227, 500]

            h50 = hits_at_k(vp.cpu(), vn.cpu(), k=50)
            scheduler.step(h50)

            if h50 > best_hits:
                best_hits = h50
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:4d} | Loss {loss.item():.4f} | "
                f"Hits@50 {h50:.4f} | Best {best_hits:.4f} | "
                f"Pat {patience_count}/{PATIENCE} | {elapsed / 60:.1f}min"
            )

            if patience_count >= PATIENCE:
                print("  Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(model, model_save_path)
    print(f"\n  Saved -> {model_save_path}  (Best Hits@50: {best_hits:.4f})")


# CLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    args = parser.parse_args()

    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(f"--task {args.task} incompatible with --dataset {args.dataset}")

    os.makedirs(args.model_dir, exist_ok=True)
    model_save_path = os.path.join(
        args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    if args.dataset == "B":
        raw_path = os.path.join(args.data_dir, "B", "data.pt")
        print("Loading dataset B with mmap...")
        data = torch.load(raw_path, map_location="cpu", mmap=True, weights_only=False)
        # No normalization — BatchNorm in model handles it per-batch

        class MmapDatasetB:
            def __getitem__(self, i):
                return data

            def __len__(self):
                return 1

        ds = MmapDatasetB()
        train_B(ds, model_save_path, device)
    else:
        ds = load_dataset(args.dataset, args.data_dir)
        if args.dataset == "A":
            train_A(ds, model_save_path, device)
        elif args.dataset == "C":
            train_C(ds, model_save_path, device)


if __name__ == "__main__":
    main()
