# foundry_t20_forge_approxabs_fail

Negative dual of foundry_t20_forge_approxabs_pass.

- `test_exceeds`: |100 - 103| = 3 > 2 → FAILED — the approximate-tolerance
  assertion catches a real out-of-tolerance discrepancy.

Verified against real forge: `[FAIL: 100 !~= 103 (max delta: 2, real delta: 3)]`.
