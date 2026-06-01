// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    uint public x;
    function setX() public virtual {
        x = 1;
    }
}

contract Derived is Base {
    function test() public {
        super.setX();
        assert(x == 0);
    }
}
