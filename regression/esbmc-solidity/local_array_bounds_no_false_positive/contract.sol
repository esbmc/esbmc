// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Guard against a bounds-check false positive: a BRACE-LESS loop body writing
// strictly in-bounds must verify clean. Before flush_pending_into_body, the
// `pos < 2` assertion emitted for `b[i]` leaked out of the loop and was checked
// with an unconstrained i, spuriously failing. With the fix the assertion stays
// under the `i < 2` guard, so k-induction proves it SUCCESSFUL.
contract MyContract {
    function run() public pure {
        uint8[2] memory b;
        for (uint i = 0; i < 2; i++)
            b[i] = 7;   // always in bounds
    }
}
// KNOWNBUG: upstream k-induction disables the inductive step here ("cannot
// havoc what a loop writes through a pointer"), so only base cases run and
// the expected proof is never reached.
