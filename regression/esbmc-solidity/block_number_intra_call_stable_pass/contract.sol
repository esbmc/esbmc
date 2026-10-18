// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.3 — positive regression coverage for block.number intra-call
// stability. Two reads of block.number in the same function must
// return the same value. Already covered by the model: block_number
// is a global symbol assigned once per harness iteration via
// _sol_per_tx_reseed; the frontend lowers block.number to a direct
// symbol read so subsequent reads see the same value.
contract H {
    function check() external view {
        uint256 a = block.number;
        uint256 b = block.number;
        assert(a == b);
    }
}
