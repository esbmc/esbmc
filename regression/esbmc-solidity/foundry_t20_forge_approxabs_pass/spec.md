# foundry_t20_forge_approxabs_pass

forge-std `assertApproxEqAbs(a, b, maxDelta)` lowered to `assert(|a-b| <= maxDelta)`.

- `test_within`:  |100 - 103| = 3 <= 5  → SUCCESSFUL (focused).
- `test_boundary`: |100 - 103| = 3 <= 3 → SUCCESSFUL (boundary inclusive).

Semantics verified against real forge 1.7.1 + forge-std v1.16.2: both PASS.
The absolute difference is encoded underflow-free as `(a>=b ? a-b : b-a)`, matching
forge-std's `stdMath.delta`.
