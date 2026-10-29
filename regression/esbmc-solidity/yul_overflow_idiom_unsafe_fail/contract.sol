// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Real-world bug pattern: a "gas-optimised" assembly add bypasses
    // Solidity 0.8+ checked arithmetic without a manual overflow check.
    // The post-condition `r >= a` would hold under checked semantics
    // (or with the lt-idiom from yul_overflow_idiom_safe_pass), but
    // the unguarded Yul add wraps silently. ESBMC must catch this.
    function check() public pure {
        uint256 a = type(uint256).max;
        uint256 b = 1;
        uint256 r;
        assembly {
            r := add(a, b)   // wraps to 0 — no revert, no panic
        }
        // r == 0, a == MAX, so r < a. Assert MUST fail.
        assert(r >= a);
    }
}
