// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pin-test: ecrecover short-circuits to nondet address and emits a
// [approx] log_warning per feedback_approx_warning_visibility.md.
// Asserting the recovered address is nondet (non-trivially) is hard;
// instead this test exercises the ecrecover path and the test.desc
// regex pins the warning emission line.
contract C {
    function rec(bytes32 h, uint8 v, bytes32 r, bytes32 s) public returns (address) {
        return ecrecover(h, v, r, s);
    }
}
