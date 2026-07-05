# Exercises the torch operational model: torch.mm computes the matrix product
# and torch.allclose confirms it equals the expected result.
import torch

X = [[1.0, 2.0], [3.0, 4.0]]
W = [[5.0, 6.0], [7.0, 8.0]]
P = torch.mm(X, W)
expected = [[19.0, 22.0], [43.0, 50.0]]
assert torch.allclose(P, expected)
