// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// COMPLEX adversary for --coverage-whole-unit: inheritance + library.
// Decisions live in three distinct declaring scopes:
//   - library L.clamp   : one `if`
//   - contract Base.setB : one `if`
//   - contract D.setD    : one `if`
//
// --contract D            (semantics A): only decisions lexically
//                          declared inside `contract D {}` count. The
//                          Base.setB `if` is attributed to Base and the
//                          L.clamp `if` to L, both excluded.
// --contract D
//   --coverage-whole-unit: D stays the harness entry, but L + Base + D
//                          decisions are all counted (whole unit).
library L {
    function clamp(uint256 v) internal pure returns (uint256) {
        if (v > 100) {
            return 100;
        }
        return v;
    }
}

contract Base {
    uint256 public b;
    function setB(uint256 w) public {
        if (w > 5) {
            b = w;
        }
    }
}

contract D is Base {
    using L for uint256;
    uint256 public d;
    function setD(uint256 v) public {
        if (v > 10) {
            d = L.clamp(v);
        }
    }
}
