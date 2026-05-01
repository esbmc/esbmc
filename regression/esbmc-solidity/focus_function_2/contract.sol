// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

// --focus-function case 1: SUCCESSFUL under bound mode.
// The constructor initializes secret = 42. f() asserts secret == 42 and
// g() clobbers it. Running with --focus-function f --contract Target --bound
// keeps constructor initialization but the nondet dispatch loop only ever
// calls f(), so the assertion holds on every path.
contract Target {
    uint public secret;

    constructor() {
        secret = 42;
    }

    function f() public view {
        assert(secret == 42);
    }

    function g() public {
        secret = 0;
    }
}
