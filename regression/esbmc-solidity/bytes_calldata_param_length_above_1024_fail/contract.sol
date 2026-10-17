// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.2 Stage S0 — KNOWNBUG pinning the [32, 1024] clamp on llc_nondet_bytes.
// Real Solidity: a `bytes calldata` parameter can have any gas-bounded length,
//                including values much greater than 1024.
// Today: solidity_builtins.c:llc_nondet_bytes() assumes length <= 1024, so the
//        assertion `d.length <= 1024` is vacuously SUCCESSFUL.
// After Stage S1 (drop the clamp): assertion produces a real counterexample
//        (e.g. length=2000), verdict flips to FAILED → rename, drop _knownbug.
contract H {
    function f(bytes calldata d) external pure {
        assert(d.length <= 1024);
    }
}
