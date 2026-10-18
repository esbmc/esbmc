// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Item 2 — cross-run idempotence (simple).
// covered.json is pre-populated with BOTH edges of the only decision.
// This run skips re-instrumenting them, but the denominator is the
// static universe (=2, Item 2c) and the numerator is credited from the
// covered-set, so coverage is bit-identical to a no-fixture run
// (Branches : 2 / Reached : 2 / 100%).
contract C {
    uint256 public x;
    function setX(uint256 v) public {
        if (v > 10) {
            x = v;
        }
    }
}
