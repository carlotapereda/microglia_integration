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
outdir = "autotune_results_100"
os.makedirs(outdir, exist_ok=True)

search_space_100 = {
    "model_params": {
        "n_hidden": tune.choice([128, 256, 512]),
        "n_layers": tune.choice([2, 3]),
        "n_latent": tune.choice([10, 20, 30, 50]),
        "dropout_rate": tune.uniform(0.1, 0.4),
    },
    "train_params": {
        "max_epochs": 100,
        "plan_kwargs": {
            "lr": tune.loguniform(1e-4, 1e-2),
            "n_epochs_kl_warmup": tune.choice([50, 100, None]),
        },
    },
}

exp = run_autotune(
    model_cls=model_cls,
    data=adata,
    metrics=["elbo_validation"],
    mode="min",
    search_space=search_space_100,
    num_samples=10,
    resources={"cpu": 6, "gpu": 1},
    log_to_driver=False
   # storage_path="ray_experiment_100",
   # name="microglia_tuning_100"
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

# ============================================================
# Round 2: 50 epochs
# ============================================================
outdir = "autotune_results_50"
os.makedirs(outdir, exist_ok=True)

search_space_50 = {
    "model_params": search_space_100["model_params"],
    "train_params": {
        "max_epochs": 50,
        "plan_kwargs": search_space_100["train_params"]["plan_kwargs"],
    },
}

exp = run_autotune(
    model_cls=model_cls,
    data=adata,
    metrics=["elbo_validation"],
    mode="min",
    search_space=search_space_50,
    num_samples=10,
    resources={"cpu": 6, "gpu": 1},
    log_to_driver=False
    #storage_path="ray_experiment_50",
   # name="microglia_tuning_50"
)

best = exp.result_grid.get_best_result(metric="elbo_validation", mode="min")
best_model_params = best.config["model_params"]
best_train_params = best.config["train_params"]

print("Round 2 - Best model params:", best_model_params)
print("Round 2 - Best train params:", best_train_params)

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
plt.title("Validation ELBO across epochs (50)")
plt.xlabel("Epoch")
plt.ylabel("Validation ELBO")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "elbo_training_curves.png"))
plt.close()

best_model = model_cls(adata, **best_model_params)
best_model.train(**best_train_params)
best_model.save(os.path.join(outdir, "best_model"), overwrite=True)

print("✅ Both rounds finished. Results saved in autotune_results_100/, autotune_results_50/")
