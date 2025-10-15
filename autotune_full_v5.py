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
# Setup
# ---------------------------
adata = sc.read_h5ad("merged_microglia.h5ad")
scvi.model.SCVI.setup_anndata(adata, batch_key="dataset")
model_cls = scvi.model.SCVI

outdir = "autotune_results_v5_balanced"
os.makedirs(outdir, exist_ok=True)

# ---------------------------
# Search space centered near your best-performing region
# ---------------------------
search_space_v5 = {
    "model_params": {
        "n_hidden": tune.choice([512, 576, 640]),
        "n_layers": tune.choice([2, 3]),
        "n_latent": tune.choice([28, 32, 36, 40, 44, 48]),
        "dropout_rate": tune.uniform(0.10, 0.18),
    },
    "train_params": {
        "max_epochs": 200,                        # early stopping will trigger sooner
        "batch_size": tune.choice([256, 512]),
        "datasplitter_kwargs": {"num_workers": 8},
        "plan_kwargs": {
            "lr": tune.loguniform(0.002, 0.004),  # focus near 0.003
            "n_epochs_kl_warmup": tune.choice([10, 15, 20, 25, 30]),
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
    search_space=search_space_v5,
    num_samples=25,
    resources={"cpu": 7, "gpu": 1},
    log_to_driver=False,
)

# ---------------------------
# Save best results
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
df.sort_values(metric_col, ascending=True).to_csv(os.path.join(outdir, "tuning_output.csv"), index=False)

# ---------------------------
# Plot ELBO curves
# ---------------------------
plt.figure(figsize=(10,6))
sns.lineplot(
    data=exp.result_grid.get_dataframe(),
    x="training_iteration",
    y=metric_col,
    hue="config/model_params/n_hidden",
    style="config/model_params/n_layers",
    alpha=0.6
)
plt.title("Validation ELBO (Balanced Search)")
plt.xlabel("Epoch")
plt.ylabel("Validation ELBO")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "elbo_training_curves.png"), dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------
# Train + save best model
# ---------------------------
best_model = model_cls(adata, **best_model_params)
best_model.train(**best_train_params)
best_model.save(os.path.join(outdir, "best_model"), overwrite=True)

# Compute UMAP for sanity check
adata.obsm["X_scvi"] = best_model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scvi")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["dataset"], show=False)
plt.savefig(os.path.join(outdir, "umap_dataset.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ Balanced tuning complete. Best model + plots saved in: {outdir}")
