// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.1 Stage S0 — CORE pass companion locking in determinism: same args
// produce the same encoding deterministically. Today and post-fix:
// SUCCESSFUL. Guards against the S1 fold-helper being implemented in a
// way that introduces fresh nondet on each call (which would break this).
contract H {
    function check(uint256 a, uint256 b) external pure {
        bytes32 h1 = keccak256(abi.encode(a, b));
        bytes32 h2 = keccak256(abi.encode(a, b));
        assert(h1 == h2);
    }
}
