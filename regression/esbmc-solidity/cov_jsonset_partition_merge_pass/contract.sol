// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Item 2 — partial covered-set, universe stable (complex).
// Run with --contract C --coverage-whole-unit, so the universe is all
// 4 edges (C.setX + Other.setY). covered.json holds ONLY C.setX's two
// edges. They are skipped this run, yet Branches stays 4 (the no-skip
// static universe — Item 2c) and Reached stays 2 (C.setX credited from
// the covered-set; Other.setY is never invoked by C's harness) =>
// Branch Coverage : 50%, identical to the no-fixture whole-unit run.
// A regressed instrumented-set denominator would instead report
// Branches : 2.
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
