# Verification harness for the torch operational model — torch.mm.
#
# Contract clause (torch.mm: 2-D matrix product A(n x k) . B(k x m) -> (n x m)):
#   E1  right identity: mm(A, I) == A for the k x k identity I. This exercises
#       the full dot-product loop (each output is a sum of A[i][t]*I[t][j]).
#
# torch tensors are float-heavy nested lists with no nondet-friendly surface,
# so this harness uses concrete anchors — the established tactic for float-heavy
# models (cf. cmath, numpy). torch.allclose is the tensor-equality oracle; a
# single mm result is cached in a local to keep the symbolic budget small.
import torch

A = [[1.0, 2.0], [3.0, 4.0]]
I = [[1.0, 0.0], [0.0, 1.0]]

# E1: A . I must equal A element-for-element.
product = torch.mm(A, I)
assert torch.allclose(product, A)
