// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.2 Stage S0 — KNOWNBUG pinning the [32, 1024] clamp on llc_nondet_bytes.
// Real Solidity: a `bytes memory` parameter can be larger than 1024 bytes.
// Today: vacuously SUCCESSFUL via the <= 1024 assume.
// After Stage S1: counterexample with length>1024 makes the assertion FAIL.
contract H {
    function g(bytes memory d) external pure {
        assert(d.length <= 1024);
    }
}
