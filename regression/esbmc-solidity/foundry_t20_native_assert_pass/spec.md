property: after `Counter c=new Counter(); c.inc()`, `c.x()==1`.
classification: correct-test
forge_truth: PASS
esbmc_expected: CORRECT (VERIFICATION SUCCESSFUL)
complexity: {cheatcodes:0, assertions:1(native), PUT:no, call-depth:1, revert-data:no, snapshot:no, cross-contract:1}
mutant-sibling: foundry_t20_native_assert_fail (assert expects x==2)
note: native `assert` (not forge-std assertEq) — proves the harness runs a test fn and a real assert surfaces. assertEq lowering is F1.b.
