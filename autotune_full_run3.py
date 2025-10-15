import scanpy as sc
import scvi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

import ray; ray.shutdown()
from scvi.autotune import run_autotune
from ray import tune

# ---------------------------
# Data + setup
# ---------------------------
adata = sc.read_h5ad("merged_microglia.h5ad")
scvi.model.SCVI.setup_anndata(adata, batch_key="dataset")
model_cls = scvi.model.SCVI

outdir = "autotune_results_v3"
os.makedirs(outdir, exist_ok=True)

# ---------------------------
# Expanded refined search space (for deeper/larger latent tuning)
# ---------------------------
search_space_v3 = {
    "model_params": {
        # more capacity to capture subtle diversity
        "n_hidden": tune.choice([512, 640, 768, 1024]),
        "n_layers": tune.choice([2, 3, 4]),
        "n_latent": tune.choice([40, 50, 64, 80]),
        "dropout_rate": tune.uniform(0.05, 0.25),
    },
    "train_params": {
        "max_epochs": 100,
        "batch_size": tune.choice([256, 512]),
        "datasplitter_kwargs": {"num_workers": 8},
        "plan_kwargs": {
            "lr": tune.loguniform(5e-4, 5e-3),
            "n_epochs_kl_warmup": tune.choice([10, 20, 40, 60]),
        },
    },
}

# ---------------------------
# Run autotune
# ---------------------------
exp = run_autotune(
    model_cls=model_cls,
    data=adata,
    metrics=["elbo_validation"],
    mode="min",
    search_space=search_space_v3,
    num_samples=35,                # broaden coverage for better exploration
    resources={"cpu": 7, "gpu": 1},
    log_to_driver=False,
)

# ---------------------------
# Get best result + parameters
# ---------------------------
best = exp.result_grid.get_best_result(metric="elbo_validation", mode="min")
best_model_params = best.config["model_params"]
best_train_params = best.config["train_params"]

print("=== Best model parameters ===")
print(best_model_params)
print("=== Best training parameters ===")
print(best_train_params)

# ---------------------------
# Save full results
# ---------------------------
df = exp.result_grid.get_dataframe()
metric_col = "metrics/elbo_validation" if "metrics/elbo_validation" in df.columns else "elbo_validation"
df.sort_values(metric_col, ascending=True).to_csv(os.path.join(outdir, "tuning_output.csv"), index=False)

# ---------------------------
# Save training histories
# ---------------------------
history_dfs = []
for result in exp.result_grid:
    hist = result.metrics_dataframe
    hist["trial_id"] = result.metrics["trial_id"]
    for k, v in result.config["model_params"].items():
        hist[k] = v
    for k, v in result.config["train_params"].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                hist[f"{k}_{kk}"] = vv
        else:
            hist[k] = v
    history_dfs.append(hist)

history_df = pd.concat(history_dfs, ignore_index=True)
history_df.to_csv(os.path.join(outdir, "training_histories.csv"), index=False)

# ---------------------------
# Plot validation ELBO curves
# ---------------------------
plt.figure(figsize=(10,6))
sns.lineplot(
    data=history_df,
    x="training_iteration",
    y=metric_col,
    hue="n_hidden",
    style="n_layers",
    estimator=None,
    units="trial_id",
    alpha=0.6
)
plt.title("Validation ELBO across epochs (Refined V3)")
plt.xlabel("Epoch")
plt.ylabel("Validation ELBO")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "elbo_training_curves.png"), dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------
# Train and save the best model
# ---------------------------
best_model = model_cls(adata, **best_model_params)
best_model.train(**best_train_params)
best_model.save(os.path.join(outdir, "best_model"), overwrite=True)

# ---------------------------
# Compute UMAP for sanity check
# ---------------------------
adata.obsm["X_scvi"] = best_model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scvi")
sc.tl.umap(adata)

# Save UMAP as figure
fig_path = os.path.join(outdir, "umap_dataset.png")
sc.pl.umap(adata, color=["dataset"], wspace=0.4, show=False)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ Tuning complete. Best model + UMAP saved to: {outdir}")
