// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.2 Stage S0 — KNOWNBUG pinning ledger entry #10 (calldata bytes[] element).
// Calldata `bytes[]` element access goes through solidity_convert_expr.cpp's
// get_index_access_expr which substitutes a fresh llc_nondet_bytes() per access.
// Real Solidity: arr[i] can have any length, including 0.
// Today: vacuous SUCCESSFUL via the >= 32 assume in llc_nondet_bytes.
// After Stage S1: counterexample with arr[0].length == 0 makes the assertion FAIL.
contract H {
    function h(bytes[] calldata arr) external pure {
        assert(arr[0].length != 0);
    }
}
