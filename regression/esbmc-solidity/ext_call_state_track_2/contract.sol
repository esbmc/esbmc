// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Negative counterpart of ext_call_state_track_1. B now exposes bump(),
// which legitimately calls a.setX(7). The bounded harness explores the
// path bump() -> g(), so a.x == 7 at g()'s assertion. ESBMC must report
// VERIFICATION FAILED -- proving that the bounded harness really does
// exercise the interleaving and that state_track_1's success is not a
// vacuous over-approximation.

contract A {
    uint x;
    address owner;

    constructor(address _owner) {
        owner = _owner;
    }

    function setX(uint _x) public {
        require(msg.sender == owner);
        x = _x;
    }

    function checkInvariant(address _expectedOwner) public view {
        assert(owner == _expectedOwner);
        // Wrong after bump(): x is 7, not 0.
        assert(x == 0);
    }
}

contract B {
    A a;

    constructor() {
        a = new A(address(this));
    }

    function bump() public {
        a.setX(7);
    }

    function g() public view {
        a.checkInvariant(address(this));
    }
}
