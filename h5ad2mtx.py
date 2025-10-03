from scipy.io import mmwrite
import scanpy as sc

adata = sc.read_h5ad("../integration/merged_microglia.h5ad")

counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
mmwrite("../integration/mergedMic_counts.mtx", counts.T)

adata.obs_names.to_series().to_csv("../integration/mergedMic_barcodes.tsv",
                                   sep="\t", index=False, header=False)
adata.var_names.to_series().to_csv("../integration/mergedMic_features.tsv",
                                   sep="\t", index=False, header=False)
adata.obs.to_csv("../integration/mergedMic_metadata.csv")
