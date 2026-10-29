property: Bank.withdraw(amt) reverts iff amt>100; test uses vm.expectRevert().
classification: bad-test
forge_truth: FAIL
esbmc_expected: WRONG (FAILED)
KEY: exercises vm.expectRevert() modeling — armed at the cheatcode, consumed by the
  next external call which asserts _ESBMC_sol_reverted_flag (reusing the existing
  __ESBMC_reverted revert-observation primitive). _pass: withdraw(200) reverts ->
  assert holds. _fail: withdraw(50) does NOT revert -> expected-revert violated -> FAILED.
  Selector/return-data payload ignored (conservative). Core Foundry idiom.
complexity: {cheatcodes:1(vm.expectRevert), assertions:0(implicit revert check), call-depth:1, revert-dep:yes}
