// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// KNOWNBUG — modifier + tuple named-returns without an explicit `return`.
// Under a modifier the body is re-scoped into an aux wrapper, so the named
// returns the body writes are DIFFERENT symbols from the outer-scope ones the
// fall-through tuple-binding sees. The tuple-binding fix is deliberately gated
// to the non-modifier path (solidity_convert_modifier.cpp), so this combination
// stays at its pre-existing broken behaviour rather than silently binding zero.
// The aqua/BalanceLib targets are modifier-free, so this gap does not affect
// them. Pinned KNOWNBUG until aux-scope named-return resolution is added.
contract H {
    modifier m() { _; }
    function f() public m returns (uint a, uint b) {
        a = 1;
        b = 2;
    }
    function check() public {
        (uint ra, uint rb) = f();
        assert(ra == 1);
        assert(rb == 2);
    }
}
