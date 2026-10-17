// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.2 Stage S0 — KNOWNBUG pinning the [32, 1024] clamp on llc_nondet_bytes.
// `bytes memory` parameters in external functions also flow through
// assign_param_nondet → llc_nondet_bytes(), so the same clamp applies.
// Real Solidity: a `bytes memory` parameter can have length 0.
// Today: vacuously SUCCESSFUL via the >= 32 assume.
// After Stage S1: counterexample with length=0 makes the assertion FAIL.
contract H {
    function g(bytes memory d) external pure {
        assert(d.length != 0);
    }
}
