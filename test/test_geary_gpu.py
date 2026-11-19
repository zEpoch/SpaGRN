"""Numerical regression against the original weighted-edge formula.

Loads the module directly so unrelated SpaGRN dependencies are not required.
Runs CPU checks everywhere and identical CUDA checks when a GPU is visible.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

path = Path(__file__).resolve().parents[1] / 'src/spagrn/global_bivariate_gearys_C_fast.py'
spec = importlib.util.spec_from_file_location('geary_under_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compute = module.global_bivariate_gearys_C_fast


class GearyRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(731)
        self.X = rng.poisson(3, (23, 11)).astype(np.float64)
        self.X[:, 9] = 0
        self.X[:, 10] = 7  # nonzero constant must also yield zero
        self.genes = pd.Index(['g%d' % i for i in range(11)])
        self.cells = pd.Index(['c%d' % i for i in range(23)])
        self.ix = rng.randint(23, size=140)
        self.iy = rng.randint(23, size=140)
        self.w = rng.uniform(.1, 1, size=140)
        self.weights = pd.DataFrame({'Cell_x': self.cells[self.ix],
                                     'Cell_y': self.cells[self.iy], 'Weight': self.w})
        self.tf = list(self.genes[:4]) + ['missing']
        self.targets = list(self.genes[4:]) + ['missing']

    def data(self, sparse=False):
        x = sp.csr_matrix(self.X) if sparse else self.X.copy()
        return SimpleNamespace(X=x, layers={'count': x},
                               var=pd.DataFrame(index=self.genes), obs_names=self.cells)

    def compare(self, device, sparse=False, batch=3):
        result = compute(self.data(sparse), self.weights, self.tf, self.targets,
                         layer_key='count', torch_device=device, pair_batch_size=batch)
        d = np.sqrt(np.maximum(((self.X-self.X.mean(0))**2).sum(0), 1e-24))
        expected = []
        for a in range(4):
            for b in range(4, 11):
                expected.append(np.sum(self.w * (self.X[self.ix,a]-self.X[self.iy,a]) *
                                       (self.X[self.ix,b]-self.X[self.iy,b])) / (d[a]*d[b]))
        np.testing.assert_allclose(result.importance, expected, rtol=1e-10, atol=1e-10)
        self.assertEqual(result.TF.tolist(), np.repeat(self.tf[:-1], 7).tolist())
        self.assertEqual(result.target.tolist(), np.tile(self.targets[:-1], 4).tolist())

    def test_cpu_dense_and_sparse_batches(self):
        for sparse in (False, True):
            for batch in (1, 3, 256):
                with self.subTest(sparse=sparse, batch=batch):
                    self.compare('cpu', sparse, batch)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_dense_and_sparse(self):
        for sparse in (False, True):
            self.compare('cuda:0', sparse)

    def test_empty_and_unmapped(self):
        for tfs in ([], ['missing']):
            out = compute(self.data(), self.weights, tfs, self.targets, torch_device='cpu', layer_key=None)
            self.assertTrue(out.empty)
            self.assertEqual(list(out.columns), ['TF', 'target', 'importance'])

    def test_invalid_batch(self):
        with self.assertRaises(ValueError):
            compute(self.data(), self.weights, self.tf, self.targets,
                    torch_device='cpu', pair_batch_size=0)

    def test_unknown_cell(self):
        self.weights.loc[0, 'Cell_x'] = 'missing'
        with self.assertRaises(KeyError):
            compute(self.data(), self.weights, self.tf, self.targets,
                    torch_device='cpu', layer_key=None)

    def test_multiple_tf_blocks(self):
        self.tf = [self.genes[i % 4] for i in range(67)]
        out = compute(self.data(), self.weights, self.tf, self.targets,
                      torch_device='cpu', layer_key='count', pair_batch_size=3)
        self.assertEqual(len(out), 67*7)
        np.testing.assert_allclose(out.importance.to_numpy().reshape(67,7)[0],
                                   out.importance.to_numpy().reshape(67,7)[64])

    def test_empty_edges(self):
        self.weights = self.weights.iloc[:0]
        result = compute(self.data(), self.weights, self.tf, self.targets,
                         torch_device='cpu', layer_key=None)
        np.testing.assert_array_equal(result.importance, 0)


if __name__ == '__main__':
    unittest.main()
