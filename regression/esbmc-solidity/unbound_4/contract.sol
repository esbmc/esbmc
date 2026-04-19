// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Ext {
    uint public x;
    function setX(uint _x) public { x = _x; }
}
contract MyContract {
    function callExt(Ext _e) public {
        _e.setX(42);
        assert(_e.x() == 42);
    }
}