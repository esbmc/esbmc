property: `vm.warp(123)` then `block.timestamp == 123`.
classification: correct-test
forge_truth: PASS
esbmc_expected: CORRECT (VERIFICATION SUCCESSFUL)
complexity: {cheatcodes:1(vm.warp, MODELED), assertions:1(native), PUT:no, call-depth:0}
status: CORE since the vm.warp/vm.roll block-env-setter modeling landed
  (handle_foundry_cheatcode, solidity_convert_call.cpp). Was the KNOWNBUG
  conservativeness anchor before that; the live anchor is now
  foundry_t20_cheatcode_unmodeled_taint_knownbug (vm.fee, unmodeled).
invocation-note: migrates to `--foundry` once the flag exists.
