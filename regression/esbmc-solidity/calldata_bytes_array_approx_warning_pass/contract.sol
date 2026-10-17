// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pin-test: calldata bytes[] element read substitutes a fresh nondet
// BytesDynamic via llc_nondet_bytes() and emits a [approx] log_warning
// per feedback_approx_warning_visibility.md.
contract C {
    function rd(bytes[] calldata a, uint i) public returns (bytes memory) {
        return a[i];
    }
}
