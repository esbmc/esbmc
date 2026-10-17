// SPDX-License-Identifier: MIT
// Regression: under --bound, a contract-typed function parameter must be
// nondet-driven through the bounded nondet-dispatch loop (reachable
// Updated State) rather than sitting at ctor defaults (Initial State).
//
// This FAIL case asserts `c.counter == 0`, which was vacuously true
// under the old IS-only behaviour (ctor sets counter=0 and no one ran
// inc() before check()). With the new bound-mode drive the solver can
// pick a dispatch trace that invokes inc(), reaching counter>=1 prior
// to check(), which makes the assertion violable.
pragma solidity >=0.8.0;

contract Counter {
    uint public counter;

    constructor() {
        counter = 0;
    }

    function inc() public {
        counter = counter + 1;
    }
}

contract Harness {
    function check(Counter c) public view {
        // Was: always-true under IS-only.
        // Now: should FAIL because `c` is driven through nondet inc().
        assert(c.counter() == 0);
    }
}
