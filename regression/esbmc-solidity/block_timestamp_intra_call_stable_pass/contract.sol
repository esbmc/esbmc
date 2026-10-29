// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.3 — positive regression coverage for block.timestamp intra-call
// stability. Same shape as block_number_intra_call_stable_pass; both
// rely on the per-tx reseed pattern in solidity_misc.c.
contract H {
    function check() external view {
        uint256 t1 = block.timestamp;
        uint256 t2 = block.timestamp;
        assert(t1 == t2);
    }
}
