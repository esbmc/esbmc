// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to stress_libsol_tuple_lhs_extcall_pass: validates that ESBMC
// still finds violations on the same code shape after the tuple-LHS
// dispatch fix. We exercise the destructure (so the fix is on the hot
// path) and then assert(false) so the verifier *must* report a
// counterexample — guards against the fix accidentally turning the
// goto program into a no-op that always verifies.

contract Pair {
    function virtualBalancesForAddition(address)
        external view returns (uint216, uint40) { return (0, 0); }
}

contract Holder {
    struct Data { uint216 balance; uint40 time; }

    Pair public pair;

    function check(address tok) public view {
        Data memory vb;
        (vb.balance, vb.time) = pair.virtualBalancesForAddition(tok);
        // Touch the destructured fields so they are not slicer-pruned.
        if (vb.balance == vb.time)
            assert(false);
        else
            assert(false);
    }
}
