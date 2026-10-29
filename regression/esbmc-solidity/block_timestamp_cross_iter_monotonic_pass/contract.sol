// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pin-test: block.timestamp must be monotone non-decreasing across
// dispatcher iterations (per reference_block_tx_monotonicity.md).
// Companion to block_number_cross_iter_monotonic_pass.
contract H {
    uint256 prev_t;
    bool seen;
    function step() public {
        if (seen) {
            assert(block.timestamp >= prev_t);
        }
        prev_t = block.timestamp;
        seen = true;
    }
}
