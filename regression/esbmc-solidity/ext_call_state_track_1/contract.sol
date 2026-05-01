// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// From Solidity SMTChecker docs (Trusted External Calls, last example).
//
// SMTChecker -- even in --model-checker-ext-calls=trusted mode --
// reports a FALSE POSITIVE on `a.x == 0` inside B.g(): its encoding
// conservatively assumes an unbounded number of outside calls to a
// can occur between transactions to B, so a.x may have been mutated
// from somewhere other than B's own entry points.
//
// ESBMC's bounded harness only dispatches B's public entry points, so
// nothing else can call a.setX. Both invariants hold:
//   (1) a.owner == address(B)
//   (2) a.x    == 0            (SMTChecker false positive)
//
// The assertions live inside A so that the state reads are direct
// inside A's own body, which is what --bound cross-contract inlining
// supports.

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
        assert(x == 0);
    }
}

contract B {
    A a;

    constructor() {
        a = new A(address(this));
    }

    function g() public view {
        a.checkInvariant(address(this));
    }
}
