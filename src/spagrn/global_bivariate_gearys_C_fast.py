"""Block sparse-Laplacian evaluation of the existing bivariate Geary score.

For the flattened (possibly directed) edge table, L = B.T @ diag(w) @ B.
The edge-product numerator equals X_tf.T @ L @ X_target. Each target block
is transformed once and reused for every TF block; no edge-by-gene dense
intermediates or full-expression densification are needed.
"""

def global_bivariate_gearys_C_fast(
    adata, weights, tfs_in_data, select_genes,
    layer_key="raw_counts", num_workers=4,
    torch_device="cuda:0", pair_batch_size=256,
):
    """Block sparse-Laplacian implementation of the existing edge-product score.

    pair_batch_size is the target block size; TF blocks are capped at 64.
    Float64 and the original per-gene denominator floor are retained.
    """
    import time
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import torch
    from tqdm.auto import tqdm

    started = time.time()
    dev = torch.device(torch_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if int(pair_batch_size) < 1:
        raise ValueError("pair_batch_size must be positive")
    names = adata.var.index
    if not names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Gene and cell names must be unique")
    tf_ids = names.get_indexer(tfs_in_data)
    tf_ids = tf_ids[tf_ids >= 0]  # original skips unmapped TFs
    target_ids = names.get_indexer(select_genes)
    target_ids = target_ids[target_ids >= 0]  # original removes unmapped targets
    nt, ny = len(tf_ids), len(target_ids)
    if not nt or not ny:
        return pd.DataFrame(columns=["TF", "target", "importance"])

    # Keep CPU input sparse. Only densify the current gene block.
    source = adata.layers[layer_key] if layer_key else adata.X
    if sp.issparse(source):
        source = source.tocsc()
    n = source.shape[0]
    ix = adata.obs_names.get_indexer(weights["Cell_x"])
    iy = adata.obs_names.get_indexer(weights["Cell_y"])
    if np.any(ix < 0) or np.any(iy < 0):
        raise KeyError("Cell names in weights do not match adata.obs_names")
    w = weights["Weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(w).all():
        raise ValueError("Non-finite edge weights")

    # L = B.T diag(w) B. Works for directed, duplicated and self edges.
    # Do NOT use D-W alone: that would change this formula for directed W.
    W = sp.coo_matrix((w, (ix, iy)), shape=(n, n)).tocsr()
    degree = np.asarray(W.sum(axis=1)).ravel() + np.asarray(W.sum(axis=0)).ravel()
    L = (sp.diags(degree) - W - W.T).tocsr()
    L.sum_duplicates()
    L.eliminate_zeros()
    with torch.no_grad():
        if hasattr(torch, "sparse_csr_tensor"):
            Lt = torch.sparse_csr_tensor(
                torch.as_tensor(L.indptr, dtype=torch.int64, device=dev),
                torch.as_tensor(L.indices, dtype=torch.int64, device=dev),
                torch.as_tensor(L.data, dtype=torch.float64, device=dev),
                size=L.shape, device=dev,
            )
        else:
            coo = L.tocoo()
            Lt = torch.sparse_coo_tensor(
                torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.int64, device=dev),
                torch.as_tensor(coo.data, dtype=torch.float64, device=dev),
                size=L.shape, device=dev,
            ).coalesce()
    del W, L

    def block(ids):
        a = source[:, ids]
        if sp.issparse(a):
            a = a.toarray()
        a = np.asarray(a, dtype=np.float64)
        # Match the original centered sum of squares, not a raw-moment shortcut.
        a = a - a.mean(axis=0)  # edge differences are invariant to per-gene shifts
        d = np.sqrt(np.maximum((a ** 2).sum(axis=0), 1e-24))
        return a, d

    target_block = int(pair_batch_size)
    tf_block = min(64, nt)
    scores = np.empty((nt, ny), dtype=np.float64)
    print("[Geary block GPU] TFs={}, targets={}, pairs={}, cells={}, target_block={}, device={}".format(
        nt, ny, nt * ny, n, target_block, dev), flush=True)
    # Transform each target block once, then compare it against ALL TF blocks.
    with torch.no_grad():
        for j in tqdm(range(0, ny, target_block), total=(ny + target_block - 1) // target_block,
                      desc="Geary target blocks"):
            y, dy = block(target_ids[j:j + target_block])
            yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float64, device=dev)
            ly = torch.sparse.mm(Lt, yt)
            del yt, y
            for i in range(0, nt, tf_block):
                x, dx = block(tf_ids[i:i + tf_block])
                xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=dev)
                numerator = (xt.T @ ly).cpu().numpy()
                scores[i:i + len(dx), j:j + len(dy)] = numerator / (dx[:, None] * dy[None, :])
                del xt, numerator, x
            del ly
    result = pd.DataFrame({
        "TF": np.repeat(np.asarray(names, dtype=object)[tf_ids], ny),
        "target": np.tile(np.asarray(names, dtype=object)[target_ids], nt),
        "importance": scores.reshape(-1),
    })
    print("[Geary block GPU] Completed {} pairs in {:.2f}s".format(len(result), time.time() - started))
    return result
