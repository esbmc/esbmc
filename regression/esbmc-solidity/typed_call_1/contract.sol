// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Signature-based dispatch for low-level .call:
// When the payload is abi.encodeWithSignature("f(T)", arg), the frontend
// must route to the exact function `f(T)` and pass the concrete argument.
// The old nondet-dispatch model would have dropped the argument and
// picked a random function, leaving x unconstrained.
contract TypedCall {
    uint256 public x;
    bool public touched;

    function setX(uint256 v) public {
        x = v;
        touched = true;
    }

    function test() public {
        x = 0;
        touched = false;
        address(this).call(abi.encodeWithSignature("setX(uint256)", uint256(42)));
        // After the call, setX must have been invoked with v == 42.
        // The helper dispatches to the static TypedCall instance, which in
        // --contract mode is the same as `this`.
        assert(touched);
        assert(x == 42);
    }
}
