// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// SIMPLE adversary for --coverage-whole-unit.
// Two independent contracts; C does NOT use Other.
//
// --contract C            (semantics A): only C.setX's decision counts
//                          => Branches : 2 (see cov_scope_sibling_*).
// --contract C
//   --coverage-whole-unit: opt out of scoping. C stays the harness
//                          entry, but the denominator spans the whole
//                          compilation unit, so Other.setY's decision
//                          is counted too => Branches : 4. Other.setY is
//                          never invoked by C's dispatcher, so its two
//                          claims stay unreached => 50%.
contract Other {
    uint256 public y;
    function setY(uint256 w) public {
        if (w > 7) {
            y = w;
        }
    }
}

contract C {
    uint256 public x;
    function setX(uint256 v) public {
        if (v > 10) {
            x = v;
        }
    }
}
