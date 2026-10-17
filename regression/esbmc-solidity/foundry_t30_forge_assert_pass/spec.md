property: Counter.inc() increments x by 1; test uses forge-std assertions.
classification: correct-test
forge_truth: PASS
esbmc_expected: CORRECT (SUCCESSFUL)
KEY: exercises forge-std assertion lowering (F1.b) — assertEq is a Test-base helper
  that normally routes through forge-std fail()+logs (invisible to ESBMC). Now lowered
  to a native assert(a==b), so a wrong-expectation test (test_eq_ok) surfaces as FAILED. This
  is what makes F1 usable on REAL Foundry tests (which never use native assert).
complexity: {cheatcodes:0, assertions:1(forge-std assertEq, LOWERED), PUT:no, call-depth:1, cross-contract:1}
mutant-sibling: the other foundry_t30_forge_assert_* (expects x==2 vs x==1)
