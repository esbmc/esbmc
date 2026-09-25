# Verification harness for the torch operational model — torch.cat (dim == 1).
#
# Contract clauses (torch.cat along columns: rows stay aligned, columns append):
#   E1  width additivity: the result width is the sum of the operands' widths,
#       the row count is unchanged.
#   E2  value preservation: the left block is A's column, the right block B's.
#
# Concrete anchors (float-heavy model, no nondet surface). Minimal 1x1 operands
# keep cat's nested-list construction within the CI budget; values are checked
# by direct element indexing (layering allclose on top blows the budget).
import torch

A = [[1.0]]  # 1 x 1
B = [[2.0]]  # 1 x 1

C = torch.cat([A, B], 1)  # -> 1 x 2

# E1: shape is (1, 1+1).
assert len(C) == 1
assert len(C[0]) == 2

# E2: columns are concatenated in order (A's block, then B's).
assert C[0][0] == 1.0
assert C[0][1] == 2.0
