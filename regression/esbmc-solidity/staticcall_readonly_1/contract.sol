// SPDX-License-Identifier: GPL-3.0
// KNOWNBUG: a re-entrant dispatch writes the `new Target()` instance through
// `t`, a heap pointer the points-to analysis cannot name, so goto_k_induction
// cannot havoc what the dispatch writes and disables the inductive step.
pragma solidity >=0.8.0 <0.9.0;

// Bound-mode staticcall read-only enforcement: the Caller snapshots
// Target.x before the staticcall and asserts it is unchanged after.
// Even though Target.modify() writes inside the nondet dispatch, the
// snapshot/restore in $staticcall#0 rolls those writes back so the
// caller observes no state change.
contract Target {
    uint public x;

    function getX() public view returns (uint) {
        return x;
    }

    function modify(uint v) public {
        x = v;
    }
}

contract Caller {
    Target t = new Target();

    function test() public {
        t.modify(100);
        uint before_x = t.x();
        (bool success, ) = address(t).staticcall(abi.encodeWithSignature("getX()"));
        uint after_x = t.x();
        assert(before_x == after_x);
        assert(success || !success);
    }
}
