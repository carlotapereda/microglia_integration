import os
import scanpy as sc
import scvi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import ray; ray.shutdown()
from scvi.autotune import run_autotune
from ray import tune

adata = sc.read_h5ad('merged_microglia.h5ad')
model_cls = scvi.model.SCVI
model_cls.setup_anndata(adata)

# ============================================================
# Round 1: 100 epochs
# ============================================================
outdir = "autotune_results_50_refined"
os.makedirs(outdir, exist_ok=True)

search_space_refined = {
    "model_params": {
        "n_hidden": tune.choice([256, 512, 768, 1024]),
        "n_layers": tune.choice([2, 3]),
        "n_latent": tune.choice([20, 30, 40, 64]),
        "dropout_rate": tune.uniform(0.1, 0.2),
    },
    "train_params": {
        "max_epochs": 50,
        "plan_kwargs": {
            "lr": tune.loguniform(1e-3, 5e-3),
            "n_epochs_kl_warmup": tune.choice([None, 20, 50]),
        },
    },
}

exp = run_autotune(
    model_cls=model_cls,
    data=adata,
    metrics=["elbo_validation"],
    mode="min",
    search_space=search_space_refined,
    num_samples=20,
    resources={"cpu": 7, "gpu": 1},
    log_to_driver=False
)

best = exp.result_grid.get_best_result(metric="elbo_validation", mode="min")
best_model_params = best.config["model_params"]
best_train_params = best.config["train_params"]

print("Round 1 - Best model params:", best_model_params)
print("Round 1 - Best train params:", best_train_params)

df = exp.result_grid.get_dataframe()
metric_col = "metrics/elbo_validation" if "metrics/elbo_validation" in df.columns else "elbo_validation"
df.sort_values(metric_col, ascending=True).to_csv(os.path.join(outdir, "tuning_output.csv"), index=False)

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
plt.title("Validation ELBO across epochs (100)")
plt.xlabel("Epoch")
plt.ylabel("Validation ELBO")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "elbo_training_curves.png"))
plt.close()

best_model = model_cls(adata, **best_model_params)
best_model.train(**best_train_params)
best_model.save(os.path.join(outdir, "best_model"), overwrite=True)

print("✅ Both rounds finished. Results saved in autotune_results_50_refined")
