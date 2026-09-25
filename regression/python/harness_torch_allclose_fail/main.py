# Negative harness for the torch operational model — torch.allclose.
#
# Pins the teeth of the E2 discrimination clause (see harness_torch_allclose):
# the plausible-but-wrong strengthening "allclose tolerates any small-looking
# discrepancy" is falsified. B differs from A by 1.0 in one cell, far above the
# default atol (1e-8) + rtol*|b| band, so allclose(A, B) is False and asserting
# it True must fail.
import torch

A = [[1.0, 2.0], [3.0, 4.0]]
B = [[1.0, 2.0], [3.0, 5.0]]

assert torch.allclose(A, B)
