# Verification harness for the torch operational model — torch.allclose.
#
# Contract clauses (torch.allclose: element-wise |a-b| <= atol + rtol*|b|):
#   E1  reflexivity: a tensor is allclose to itself.
#   E2  discrimination: tensors differing by more than the tolerance are NOT
#       allclose (the check is not the constant True).
#
# Concrete anchors (float-heavy model, no nondet surface). The E2 anchor differs
# by 1.0 in one cell, far above the default atol (1e-8) + rtol*|b| band.
import torch

A = [[1.0, 2.0], [3.0, 4.0]]
B = [[1.0, 2.0], [3.0, 5.0]]

# E1: reflexive.
assert torch.allclose(A, A)

# E2: a 1.0 discrepancy is rejected.
assert not torch.allclose(A, B)
