import os
import scanpy as sc
import scvi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import ray; ray.shutdown()
from scvi.autotune import run_autotune
from ray import tune

# ---------------------------
# Setup
# ---------------------------
adata = sc.read_h5ad("merged_microglia_fixed.h5ad")
scvi.model.SCVI.setup_anndata(adata, batch_key="dataset")
model_cls = scvi.model.SCVI

outdir = "autotune_results_focus_v6"
os.makedirs(outdir, exist_ok=True)

# ---------------------------
# Focused search space near your best-performing region
# (tight ranges; emphasize lr≈0.003 and typically no KL warmup)
# ---------------------------
search_space_focus = {
    "model_params": {
        "n_hidden": tune.choice([512, 576, 640]),
        "n_layers": tune.choice([2, 3]),
        "n_latent": tune.choice([28, 30, 32, 36]),
        "dropout_rate": tune.uniform(0.12, 0.18),
        # fixed likelihood per your runs
        "gene_likelihood": "nb",
    },
    "train_params": {
        "max_epochs": 200,                     # early stopping will cut it short
        "batch_size": tune.choice([256, 512]),
        "datasplitter_kwargs": {"num_workers": 8},
        "plan_kwargs": {
            "lr": tune.loguniform(0.0027, 0.0033),
            "n_epochs_kl_warmup": tune.choice([None, 10, 15]),
        },
        # add early stopping + GPU settings directly to SCVI.train kwargs
        "early_stopping": True,
        "early_stopping_patience": 25,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
    },
}

# ---------------------------
# Run autotune (compact trial budget)
# ---------------------------
exp = run_autotune(
    model_cls=model_cls,
    data=adata,
    metrics=["elbo_validation"],   # if you later want to optimize for mixing, we can add a custom metric
    mode="min",
    search_space=search_space_focus,
    num_samples=28,                # ~24–32 recommended; using 28 here
    resources={"cpu": 7, "gpu": 1},
    log_to_driver=False,
)

# ---------------------------
# Save best result + a ranked CSV of all trials
# ---------------------------
best = exp.result_grid.get_best_result(metric="elbo_validation", mode="min")
best_model_params = best.config["model_params"]
best_train_params = best.config["train_params"]

print("=== Best model parameters ===")
print(best_model_params)
print("=== Best training parameters ===")
print(best_train_params)

df = exp.result_grid.get_dataframe()
metric_col = "metrics/elbo_validation" if "metrics/elbo_validation" in df.columns else "elbo_validation"
df_sorted = df.sort_values(metric_col, ascending=True)
df_sorted.to_csv(os.path.join(outdir, "tuning_output.csv"), index=False)

# Also save top-5 for quick inspection in logs
print("\nTop-5 trials by ELBO:")
print(df_sorted[[metric_col, "config/model_params/n_hidden",
                 "config/model_params/n_layers",
                 "config/model_params/n_latent",
                 "config/train_params/plan_kwargs/n_epochs_kl_warmup",
                 "config/train_params/batch_size"]].head(5))

# ---------------------------
# Plot ELBO curves
# ---------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="training_iteration",
    y=metric_col,
    hue="config/model_params/n_hidden",
    style="config/model_params/n_layers",
    alpha=0.6
)
plt.title("Validation ELBO (Focused Search)")
plt.xlabel("Epoch")
plt.ylabel("Validation ELBO")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "elbo_training_curves.png"), dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------
# Train + save the best model on full data
# ---------------------------
best_model = model_cls(adata, **best_model_params)
best_model.train(**best_train_params)
best_model.save(os.path.join(outdir, "best_model"), overwrite=True)

# ---------------------------
# UMAP sanity check
# ---------------------------
adata.obsm["X_scvi"] = best_model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scvi")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["dataset"], show=False)
plt.savefig(os.path.join(outdir, "umap_dataset.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ Focused tuning complete. Best model + plots saved in: {outdir}")
