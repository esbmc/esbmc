// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Item 2e — monotone / crash-safe tripwire (deterministic, no timing).
// Under --contract C --coverage-whole-unit the universe is 4 edges.
// covered.json is seeded with ALL 4, including Other.setY's two — which
// are UNREACHABLE under C's harness, so the run can never re-witness
// them. They are credited only because the persisted covered-set is
// monotone (load never drops, atomic write never truncates):
//   Branches : 4 / Reached : 4 / 100%.
// A regression that loses a committed edge (non-monotone load, or a
// non-atomic / truncating write) would drop the 2 unreachable seeds =>
// Reached : 2 / 50%. The 100%/Reached:4 pin is that tripwire.
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
