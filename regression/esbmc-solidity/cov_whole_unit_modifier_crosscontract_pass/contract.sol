// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// ADVERSARIAL for --coverage-whole-unit: it is the explicit escape
// hatch that recovers the named semantics-A completeness trade-off.
//
// A declares a modifier `gate` (its body has a branch) and a branchy
// internal `bumpInternal`. B inherits A and applies `gate` to setB.
//
// --contract B            (semantics A): the `gate` and `bumpInternal`
//                          decisions are textually declared inside A, so
//                          they are attributed to A and excluded. B.setB
//                          has no own decision => "No branch detected".
//                          (This is the accepted 3b trade-off, pinned by
//                          cov_scope_modifier_crosscontract_knownbug.)
// --contract B
//   --coverage-whole-unit: opt out of scoping. The A-declared decisions
//                          applied/inherited by B are counted again, so
//                          coverage is no longer silently zero.
contract A {
    uint256 public a;
    modifier gate(uint256 z) {
        if (z > 100) {
            _;
        }
    }
    function bumpInternal(uint256 p) internal {
        if (p > 3) {
            a = p;
        }
    }
}

contract B is A {
    uint256 public b;
    function setB(uint256 v) public gate(v) {
        b = v;
    }
}
