// SPDX-License-Identifier: MIT
// Regression: under --bound, a contract-typed function parameter is
// driven through the bounded nondet-dispatch loop.  This PASS case
// asserts a property that holds for every reachable state (unsigned
// counter is always >= 0), so the drive must not cause a spurious
// violation.
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
        // True regardless of whether `c` stayed at IS or was driven to
        // any reachable US — `counter` is unsigned and inc() only adds.
        assert(c.counter() >= 0);
    }
}
