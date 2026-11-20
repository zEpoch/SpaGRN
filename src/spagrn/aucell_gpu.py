"""Cell-blocked PyTorch AUCell, matching pySCENIC 0.12.1 / ctxcore 0.2.0.

Stable sorting after the SAME pandas column permutation preserves tie semantics.
The weighted AUC is sum_g w_g * max(cutoff-rank_g, 0) / ((cutoff+1)*sum(w)).
"""
import logging

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def aucell_gpu(exp_mtx, signatures, auc_threshold=0.05, noweights=False,
               normalize=False, seed=None, torch_device='cuda:0', cell_batch_size=512):
    from ctxcore.recovery import derive_rank_cutoff

    if not isinstance(exp_mtx, pd.DataFrame) or not exp_mtx.columns.is_unique:
        raise ValueError('AUCell requires a DataFrame with unique gene columns')
    if int(cell_batch_size) < 1:
        raise ValueError('cell_batch_size must be positive')
    device = torch.device(torch_device)
    signatures = list(signatures)
    if len({m.name for m in signatures}) != len(signatures):
        raise ValueError('AUCell requires unique signature names')
    n, g = exp_mtx.shape
    cutoff = derive_rank_cutoff(auc_threshold, g)
    names = pd.Index([s.name for s in signatures], name='Regulon')
    cells = exp_mtx.index.rename('Cell')
    if not signatures:
        return pd.DataFrame(index=cells, columns=names, dtype=float)
    # Matches DataFrame.sample(axis=1, random_state=seed) without copying expression.
    perm = pd.Series(np.arange(g)).sample(frac=1, random_state=seed).to_numpy()
    genes = exp_mtx.columns[perm]
    rows, cols, vals = [], [], []
    for i, signature in enumerate(signatures):
        ids = np.flatnonzero(genes.isin(signature.genes))
        if not len(ids) or len(ids) / float(len(signature)) < .8:
            LOGGER.warning('Less than 80%% of genes in %s are present; returning zero AUC', signature.name)
            continue
        w = np.ones(len(ids)) if noweights else np.array([signature[genes[j]] for j in ids], dtype=float)
        maxauc = (cutoff + 1) * w.sum()
        if not maxauc > 0:
            raise ValueError('AUCell signature weights must have a positive finite sum')
        rows.extend([i] * len(ids))
        cols.extend(ids.tolist())
        vals.extend((w / maxauc).tolist())
    weights = torch.sparse_coo_tensor(
        torch.as_tensor(np.array([rows, cols]), dtype=torch.int64, device=device),
        torch.as_tensor(vals, dtype=torch.float64, device=device),
        size=(len(signatures), g), device=device,
    ).coalesce()
    out = np.empty((n, len(signatures)), dtype=np.float64)
    print('[AUCell GPU] cells={}, genes={}, signatures={}, block={}, device={}'.format(
        n, g, len(signatures), cell_batch_size, device), flush=True)
    with torch.no_grad():
        for start in tqdm(range(0, n, int(cell_batch_size)), desc='AUCell cell blocks'):
            end = min(n, start + int(cell_batch_size))
            a = exp_mtx.iloc[start:end, perm].to_numpy(dtype=np.float64)
            x = torch.as_tensor(np.ascontiguousarray(a), device=device)
            missing = torch.isnan(x)
            order = torch.argsort(x.masked_fill(missing, -float('inf')), dim=1,
                                  descending=True, stable=True)
            # Pandas na_option='bottom': NaNs come AFTER genuine -inf values.
            if bool(missing.any()):
                flags = missing.gather(1, order).to(torch.int8)
                order = order.gather(1, torch.argsort(flags, dim=1, stable=True))
            selected = order[:, :cutoff]
            scores = torch.zeros_like(x)
            contribution = torch.arange(cutoff, 0, -1, device=device, dtype=torch.float64)
            scores.scatter_(1, selected, contribution.expand(end-start, -1))
            out[start:end] = torch.sparse.mm(weights, scores.T).T.cpu().numpy()
            del x, missing, order, selected, scores
    result = pd.DataFrame(out, index=cells, columns=names)
    return result / result.max(axis=0) if normalize else result
