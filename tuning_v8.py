import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
adata = sc.read_h5ad("merged_microglia_fixed.h5ad")
scvi.model.SCVI.setup_anndata(adata, batch_key="dataset")
model_cls = scvi.model.SCVI
BATCH_KEY = "dataset"

df = pd.read_csv("tuning_output_elbo.csv")

# Optional: only rescore top-N by ELBO to save time
# df = df.nsmallest(10, "elbo_validation")

# ------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------
def knn_entropy(X, batches, k=50):
    """Normalized kNN entropy for batch mixing (higher=better)."""
    nbrs = NearestNeighbors(n_neighbors=min(k+1, X.shape[0])).fit(X)
    idx = nbrs.kneighbors(X, return_distance=False)[:, 1:]
    nb = batches[idx]
    cats = pd.Categorical(batches).categories
    cat2idx = {c: i for i, c in enumerate(cats)}
    nb_codes = np.vectorize(cat2idx.get)(nb)
    B = len(cats)
    logB = np.log(B) if B > 1 else 1.0
    hist = np.zeros((nb_codes.shape[0], B))
    for i in range(B):
        hist[:, i] = (nb_codes == i).sum(axis=1)
    p = hist / hist.sum(axis=1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1) / logB
    return float(ent.mean())

def silhouette_score_latent(X, batches):
    """Mean silhouette per batch label (closer to 0 = better mixing)."""
    sil = silhouette_samples(X, batches)
    return float(np.mean(sil))

# ------------------------------------------------------------
# Loop through trials and compute both metrics
# ------------------------------------------------------------
results = []
for i, row in df.iterrows():
    model_params = {
        "n_hidden": int(row["config/model_params/n_hidden"]),
        "n_layers": int(row["config/model_params/n_layers"]),
        "n_latent": int(row["config/model_params/n_latent"]),
        "dropout_rate": float(row["config/model_params/dropout_rate"]),
        "gene_likelihood": "nb",
    }
    train_params = {
        "max_epochs": 200,
        "batch_size": int(row["config/train_params/batch_size"]),
        "datasplitter_kwargs": {"num_workers": 8},
        "plan_kwargs": {
            "lr": float(row["config/train_params/plan_kwargs/lr"]),
            "n_epochs_kl_warmup": None
            if pd.isna(row["config/train_params/plan_kwargs/n_epochs_kl_warmup"])
            else int(row["config/train_params/plan_kwargs/n_epochs_kl_warmup"]),
        },
        "early_stopping": True,
        "early_stopping_patience": 25,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
    }

    print(f"▶️ Trial {i+1}/{len(df)} | Layers={model_params['n_layers']} "
          f"Latent={model_params['n_latent']} "
          f"LR={train_params['plan_kwargs']['lr']:.5f}")

    model = model_cls(adata, **model_params)
    model.train(**train_params)

    X = model.get_latent_representation()
    entropy_val = knn_entropy(X, adata.obs[BATCH_KEY].to_numpy(), k=50)
    silhouette_val = silhouette_score_latent(X, adata.obs[BATCH_KEY].to_numpy())

    results.append({
        "trial_id": row["trial_id"],
        "n_layers": model_params["n_layers"],
        "n_latent": model_params["n_latent"],
        "dropout_rate": model_params["dropout_rate"],
        "lr": train_params["plan_kwargs"]["lr"],
        "n_epochs_kl_warmup": train_params["plan_kwargs"]["n_epochs_kl_warmup"],
        "entropy": entropy_val,
        "silhouette": silhouette_val,
        "elbo_validation": row["elbo_validation"],
    })

    print(f"   → Entropy = {entropy_val:.4f}, Silhouette = {silhouette_val:.4f}")

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df = metrics_df.sort_values("entropy", ascending=False)
metrics_df.to_csv("tuning_output_entropy_silhouette.csv", index=False)

print("\n✅ Saved tuning_output_entropy_silhouette.csv (ranked by entropy).")

best = metrics_df.iloc[0]
print("\n=== Best-by-entropy config ===")
print(best)
