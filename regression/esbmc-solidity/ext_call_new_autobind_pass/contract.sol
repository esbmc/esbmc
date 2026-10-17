// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test that cross-contract calls on new-created instances execute the callee
// body even WITHOUT --bound. After e.setX(42), the state x inside e must
// equal 42, so checkX(42) should succeed.
//
// Before the auto-bind fix, unbound mode modeled e.setX(42) as nondet
// (no side effects) and e.checkX(42) as nondet (assertion never reached),
// producing 0 VCCs — a silent false negative.

contract Ext {
    uint x;

    function setX(uint _x) public { x = _x; }

    function checkX(uint _expected) public view {
        assert(x == _expected);
    }
}

contract Caller {
    Ext e = new Ext();

    function callExt() public {
        e.setX(42);
        e.checkX(42);
    }
}
