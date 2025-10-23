import os
import scanpy as sc
import scvi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples

from scvi.autotune import run_autotune
from ray import tune

# ------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------
adata = sc.read_h5ad("merged_microglia_fixed.h5ad")
scvi.model.SCVI.setup_anndata(adata, batch_key="dataset")
model_cls = scvi.model.SCVI
BATCH_KEY = "dataset"

outdir = "autotune_results_entropy_v8"
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------------
# 2. Define stable search space
# ------------------------------------------------------------
search_space_focus = {
    "model_params": {
        "n_hidden": tune.choice([512]),          # fixed at 512
        "n_layers": tune.choice([2, 3]),
        "n_latent": tune.choice([30, 32]),
        "dropout_rate": tune.uniform(0.14, 0.18),
        "gene_likelihood": "nb",
    },
    "train_params": {
        "max_epochs": 200,
        "batch_size": tune.choice([256]),
        "datasplitter_kwargs": {"num_workers": 8},
        "plan_kwargs": {
            "lr": tune.loguniform(0.001, 0.003),
            "n_epochs_kl_warmup": tune.choice([None, 10, 12, 15]),
        },
        "early_stopping": True,
        "early_stopping_patience": 25,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
    },
}

# ------------------------------------------------------------
# 3. Run autotune (ELBO-based, stable)
# ------------------------------------------------------------
exp = run_autotune(
    model_cls=model_cls,
    data=adata,
    metrics=["elbo_validation"],
    mode="min",
    search_space=search_space_focus,
    num_samples=24,
    resources={"cpu": 7, "gpu": 1},
    log_to_driver=False,
)

# Save ELBO results
df = exp.result_grid.get_dataframe()
metric_col = "metrics/elbo_validation" if "metrics/elbo_validation" in df.columns else "elbo_validation"
df.sort_values(metric_col, ascending=True).to_csv(os.path.join(outdir, "tuning_output_elbo.csv"), index=False)

print(f"✅ Finished run_autotune — now re-evaluating by kNN entropy...")

# ------------------------------------------------------------
# 4. Define kNN entropy metric
# ------------------------------------------------------------
def knn_entropy(X, batches, k=50):
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

# ------------------------------------------------------------
# 5. Re-score each model by entropy
# ------------------------------------------------------------
scores = []
for i, result in enumerate(exp.result_grid):
    cfg_m = result.config["model_params"]
    cfg_t = result.config["train_params"]

    print(f"▶️ Re-training model {i+1}/{len(exp.result_grid)}: "
          f"layers={cfg_m['n_layers']}, latent={cfg_m['n_latent']}, "
          f"dropout={cfg_m['dropout_rate']:.3f}, lr={cfg_t['plan_kwargs']['lr']:.5f}")

    model = model_cls(adata, **cfg_m)
    model.train(**cfg_t)
    X = model.get_latent_representation()
    entropy_val = knn_entropy(X, adata.obs[BATCH_KEY].to_numpy(), k=50)
    scores.append({
        "trial_id": i,
        "n_layers": cfg_m["n_layers"],
        "n_latent": cfg_m["n_latent"],
        "dropout_rate": cfg_m["dropout_rate"],
        "lr": cfg_t["plan_kwargs"]["lr"],
        "n_epochs_kl_warmup": cfg_t["plan_kwargs"]["n_epochs_kl_warmup"],
        "entropy": entropy_val
    })
    print(f"   → Entropy = {entropy_val:.4f}")

entropy_df = pd.DataFrame(scores).sort_values("entropy", ascending=False)
entropy_df.to_csv(os.path.join(outdir, "tuning_output_entropy.csv"), index=False)

best_conf = entropy_df.iloc[0]
print("\n=== Best-by-entropy config ===")
print(best_conf)

# ------------------------------------------------------------
# 6. Retrain & evaluate best model by silhouette + entropy
# ------------------------------------------------------------
best_model_params = {
    "n_hidden": 512,
    "n_layers": int(best_conf["n_layers"]),
    "n_latent": int(best_conf["n_latent"]),
    "dropout_rate": float(best_conf["dropout_rate"]),
    "gene_likelihood": "nb",
}
best_train_params = {
    "max_epochs": 200,
    "batch_size": 256,
    "datasplitter_kwargs": {"num_workers": 8},
    "plan_kwargs": {
        "lr": float(best_conf["lr"]),
        "n_epochs_kl_warmup": None if str(best_conf["n_epochs_kl_warmup"]) == "None" else int(best_conf["n_epochs_kl_warmup"]),
    },
    "early_stopping": True,
    "early_stopping_patience": 25,
    "accelerator": "gpu",
    "devices": 1,
    "precision": "16-mixed",
}

best_model = model_cls(adata, **best_model_params)
best_model.train(**best_train_params)
X_latent = best_model.get_latent_representation()
entropy_val = knn_entropy(X_latent, adata.obs[BATCH_KEY].to_numpy(), k=50)
silhouette_val = float(np.mean(silhouette_samples(X_latent, adata.obs[BATCH_KEY])))

print("\n✅ Final metrics for best model:")
print(f"[kNN Entropy] Mean normalized entropy: {entropy_val:.4f}")
print(f"[Silhouette] Mean per-cell silhouette: {silhouette_val:.4f}")

best_model.save(os.path.join(outdir, "best_model_entropy"), overwrite=True)
entropy_df.to_csv(os.path.join(outdir, "all_entropy_scores.csv"), index=False)

print(f"\n✅ Entropy-based tuning complete. Results saved in: {outdir}")
