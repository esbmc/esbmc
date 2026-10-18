// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression for the OTHER side of the selective-wrap heuristic.
//
// Neither addA nor addB has an explicit `require` / `revert` / `assert`.
// They both just increment `counter` by 1.  The harness generator must
// NOT wrap them in `try/catch` — ESBMC's `TryStatement` modelling has a
// nondet "catch arm" that represents "the call did not happen", and
// under this model two calls that would always succeed in real EVM gain
// phantom paths where either/both are skipped.  For addition these
// phantom paths produce different final counters across orderings (e.g.
// c1 skipped both, c2 ran both → c1.counter = 0, c2.counter = 2) and a
// naive always-wrap harness would spuriously report TOD.
//
// The selective-wrap heuristic keeps calls raw when the callee has no
// syntactic revert construct, so this test must verify CLEAN.
//
// Final states:
//   Order 1: addA(); addB(); → counter = 2
//   Order 2: addB(); addA(); → counter = 2
//   (commutative, no race)
contract Counter {
    uint public counter;

    constructor() {
        counter = 0;
    }

    function addA() public {
        counter = counter + 1;
    }

    function addB() public {
        counter = counter + 1;
    }
}
