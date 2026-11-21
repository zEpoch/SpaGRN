"""Optional single-process GPU numerical backend for pySCENIC 0.12.1.

Database reads, annotation joins, leading edges and Regulon construction retain
pySCENIC code. A private function clone replaces ONLY calc_aucs and recovery;
no global monkeypatching and no CUDA multiprocessing/forking is used.
"""
import types
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


class _DeviceCurves:
    """Keep ALL motif curves on device; copy only retained motifs to the CPU."""
    def __init__(self, tensor):
        self.tensor = tensor

    def mean(self, axis=0):
        return self.tensor.mean(dim=axis).cpu().numpy()

    def std(self, axis=0):
        return self.tensor.std(dim=axis, unbiased=False).cpu().numpy()

    def __getitem__(self, key):
        def convert(k):
            return torch.as_tensor(k, device=self.tensor.device) if isinstance(k, np.ndarray) else k
        key = tuple(convert(k) for k in key) if isinstance(key, tuple) else convert(key)
        return _DeviceCurves(self.tensor[key])


class RecoveryGPU:
    def __init__(self, device):
        self.device = torch.device(device)

    def ranks(self, df):
        cached = df.attrs.get('_spagrn_gpu_ranks')
        if cached is not None:
            return cached
        return torch.as_tensor(np.ascontiguousarray(df.values), dtype=torch.int64, device=self.device)

    def aucs(self, df, total_genes, weights, auc_threshold):
        from ctxcore.recovery import derive_rank_cutoff
        cutoff = derive_rank_cutoff(auc_threshold, total_genes)
        maxauc = float((cutoff + 1) * np.sum(weights))
        if not maxauc > 0:
            raise ValueError('Non-positive maximum AUC')
        ranks = self.ranks(df)
        w = torch.as_tensor(weights, dtype=torch.float64, device=self.device)
        result = torch.empty(len(df), dtype=torch.float64, device=self.device)
        for i in range(0, len(df), 2048):
            r = ranks[i:i+2048]
            result[i:i+2048] = ((cutoff - r).clamp_min(0).to(torch.float64) * w).sum(1) / maxauc
        return result.cpu().numpy()

    def recovery(self, df, total_genes, weights, rank_threshold, auc_threshold, no_auc=False):
        from ctxcore.recovery import derive_rank_cutoff
        cutoff = derive_rank_cutoff(auc_threshold, total_genes, rank_threshold)
        ranks = self.ranks(df).to(torch.int64)
        # Extra bin receives all ranks outside the recovery window.
        bins = torch.zeros((len(df), rank_threshold+1), dtype=torch.float64, device=self.device)
        w = torch.as_tensor(weights, dtype=torch.float64, device=self.device)
        bins.scatter_add_(1, ranks.clamp(max=rank_threshold), w.expand(len(df), -1))
        curves = bins[:, :rank_threshold].cumsum(1)
        aucs = np.array([]) if no_auc else (
            curves[:, :cutoff].sum(1) / float((cutoff+1)*np.sum(weights))).cpu().numpy()
        return _DeviceCurves(curves), aucs


class _CachedDatabase:
    def __init__(self, db, modules, device):
        from ctxcore.genesig import GeneSignature
        self.name, self.total_genes = db.name, db.total_genes
        union = set(g for module in modules for g in module.genes)
        self.frame = db.load(GeneSignature(name='spagrn_gpu_union', gene2weight={g: 1. for g in union}))
        self.genes = self.frame.columns
        # One database at a time. Reject oversized databases before allocating CUDA memory.
        required = self.frame.size * 4
        print('[cisTarget GPU] caching {} motifs x {} genes ({:.2f} GiB device ranks)'.format(
            len(self.frame), len(self.genes), required / 2**30), flush=True)
        if torch.device(device).type == 'cuda':
            free, _ = torch.cuda.mem_get_info(torch.device(device))
            if required > free * .6:
                raise MemoryError('Ranking cache needs {:.1f} GiB; reduce database scope or use CPU backend'.format(required/2**30))
        self.tensor = torch.as_tensor(np.ascontiguousarray(self.frame.values), dtype=torch.int32, device=device)

    def load(self, module):
        ids = np.flatnonzero(self.genes.isin(module.genes))
        df = self.frame.iloc[:, ids].copy()
        idx = torch.as_tensor(ids, device=self.tensor.device, dtype=torch.int64)
        df.attrs['_spagrn_gpu_ranks'] = self.tensor.index_select(1, idx)
        return df


def prune2df_gpu(rnkdbs, modules, motif_annotations_fname, torch_device='cuda:0',
                 rank_threshold=1500, auc_threshold=.05, nes_threshold=3.,
                 motif_similarity_fdr=.001, orthologuous_identity_threshold=0.,
                 weighted_recovery=False, filter_for_annotation=True):
    """Return the standard motifs dataframe; GPU AUC/recovery, original leading edges.

    CPU device is accepted only for numerical regression/debugging.
    This backend currently requires EXACT pySCENIC 0.12.1.
    """
    if version('pyscenic') != '0.12.1':
        raise RuntimeError('GPU cisTarget backend requires pySCENIC 0.12.1')
    from pyscenic.transform import module2features_auc1st_impl, module2df, DF_META_DATA
    from pyscenic.utils import load_motif_annotations

    annotations = load_motif_annotations(motif_annotations_fname,
        motif_similarity_fdr=motif_similarity_fdr,
        orthologous_identity_threshold=orthologuous_identity_threshold)
    backend = RecoveryGPU(torch_device)
    original = module2features_auc1st_impl
    namespace = dict(original.__globals__, calc_aucs=backend.aucs, recovery=backend.recovery)
    numerical = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)

    def features(*args, **kwargs):
        result = list(numerical(*args, rank_threshold=rank_threshold, auc_threshold=auc_threshold,
                               nes_threshold=nes_threshold, filter_for_annotation=filter_for_annotation, **kwargs))
        if isinstance(result[1], _DeviceCurves):
            result[1] = result[1].tensor.cpu().numpy()
        return tuple(result)

    modules, rnkdbs = list(modules), list(rnkdbs)
    results = []
    print('[cisTarget GPU] databases={}, modules={}, device={}; GPU AUC/recovery, CPU annotation/leading edges'.format(
        len(rnkdbs), len(modules), torch_device), flush=True)
    with torch.no_grad(), tqdm(total=len(modules)*len(rnkdbs), desc='cisTarget GPU modules') as progress:
        for db in rnkdbs:
            genes = set(db.genes)
            # Same >=20% missing rejection as module2df, performed BEFORE expensive enrichment.
            valid = [m for m in modules if len(m) and (len(m)-len(set(m.genes)&genes))/float(len(m)) < .20]
            progress.update(len(modules)-len(valid))
            print('[cisTarget GPU] {}: eligible={}, skipped={}'.format(db.name, len(valid), len(modules)-len(valid)), flush=True)
            if not valid:
                continue
            cached = _CachedDatabase(db, valid, torch_device)
            for module in valid:
                result = module2df(cached, module, annotations, weighted_recovery=weighted_recovery,
                                   module2features_func=features)
                if not result.empty:
                    results.append(result)
                progress.update(1)
            del cached
    return pd.concat(results) if results else DF_META_DATA.copy()
