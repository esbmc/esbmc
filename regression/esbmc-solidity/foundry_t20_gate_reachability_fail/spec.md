property: a real bug (`assert(1==2)`) precedes an unmodeled cheatcode (`vm.fee`).
classification: bad-test (the pre-cheatcode assertion is genuinely violated)
forge_truth: FAIL
esbmc_expected: WRONG (VERIFICATION FAILED)
GATE REACHABILITY CONTROL (Codex #2): proves the hard-taint prune is
  reachability-sensitive — it suppresses only the suffix AFTER the reached
  unmodeled cheatcode, NOT assertions before it. A pre-cheatcode bug must
  still surface as FAILED, else the gate would hide real bugs (a lazy
  taint-and-prune-everything implementation would wrongly report SUCCESSFUL).
complexity: {cheatcodes:1(vm.fee, UNMODELED), assertions:1(native, BEFORE cheatcode)}
