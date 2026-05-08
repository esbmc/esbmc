// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// From Solidity SMTChecker docs (Trusted External Calls).
//
// SMTChecker's default (untrusted) mode reports a false warning on the
// post-call assertion because `e` could hold any contract. SMTChecker's
// --model-checker-ext-calls=trusted mode silences that warning, but the
// docs explicitly note that trusted mode can become unsound across
// contract-type casts.
//
// ESBMC's --bound mode inlines cross-contract calls against the
// compile-time source. This is sound AND precise: after e.setX(42),
// the state x inside e must equal 42. We express the post-condition as
// a view method inside Ext so that the assertion reads Ext's own state
// directly.

contract Ext {
    uint x;

    function setX(uint _x) public { x = _x; }

    function checkX(uint _expected) public view {
        assert(x == _expected);
    }
}

contract MyContract {
    Ext e = new Ext();

    function callExt() public {
        e.setX(42);
        e.checkX(42);
    }
}
