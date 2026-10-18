property: assertEqDecimal/assertEqUint/assertGtDecimal verdict == base comparison.
forge_truth: PASS (VERIFIED vs real forge v1.16.2)
esbmc_expected: SUCCESSFUL
KEY: *Decimal variants ignore the decimals arg for the verdict (only formats the
  failure message); assertEqUint == assertEq. Conformance-verified 6/6 vs forge.
