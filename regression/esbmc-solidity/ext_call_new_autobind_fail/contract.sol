// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Negative counterpart of ext_call_new_autobind_pass. After e.setX(42),
// checkX(41) should FAIL because 42 != 41. Without auto-bind, unbound
// mode never executes the callee body, so the assertion is unreachable
// and the tool vacuously reports SUCCESS — a false negative.

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
        e.checkX(41);
    }
}
