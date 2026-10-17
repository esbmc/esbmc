property: `vm.fee(7)` then `block.basefee == 7`, where vm.fee is UNMODELED.
classification: correct-test
forge_truth: PASS
esbmc_expected: CORRECT (VERIFICATION SUCCESSFUL)
GATE TEST: vm.fee is an unmodeled cheatcode. The conservative hard-taint gate
  (handle_foundry_cheatcode fall-through) lowers it to ASSUME(false), pruning
  the un-modelable continuation, so no FALSE VERIFICATION FAILED is emitted.
  Was KNOWNBUG (false FAILED) before the gate landed.
complexity: {cheatcodes:1(vm.fee, UNMODELED->pruned), assertions:1(native, downstream->pruned)}
