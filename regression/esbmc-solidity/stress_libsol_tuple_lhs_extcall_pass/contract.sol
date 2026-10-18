// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for the `migrate expr failed` crash when destructuring a
// tuple-returning external call into struct-field LHS targets.
//
// The frontend builds a code_blockt to hold the LHS member exprs. The
// external-call RHS is rewritten by convert_unboundcall_nondet into a
// plain `sideeffect/nondet`, which loses its TUPLE_RETURNS tag. Without
// the dispatch fix, the standard assign path then embeds the LHS code
// block as an operand of an `assign` side-effect, which trips
// `migrate expr failed` in the goto layer. With the fix, the tuple
// destructuring path takes over and assigns an independent nondet
// value to each LHS slot.

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
        // Each slot is independent nondet, so this is provably true:
        // (vb.balance == vb.balance).
        assert(vb.balance == vb.balance);
    }
}
