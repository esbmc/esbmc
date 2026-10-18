// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// A local fixed-size memory array `T[k]` is also pointer-backed (length kept
// only as `#sol_array_size` metadata). get_index_access_expr turns that into a
// `pos < k` claim. The access sits in a BRACE-LESS for-body, which exercises
// flush_pending_into_body: the bounds assertion must stay under the loop guard
// (`i < 3`) so the i == 2 iteration is the one that violates `i < 2`.
// Regression pin for the local pointer-backed array OOB fix (fixed path) AND
// the brace-less loop-body path-condition placement.
contract MyContract {
    function run() public pure {
        uint8[2] memory b;
        for (uint i = 1; i < 3; i++)
            b[i] = 100;   // OUT OF BOUNDS at i == 2 (length is 2)
    }
}
