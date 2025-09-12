// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Base {
    mapping(uint => uint) public x;
    function func() public {}
}

contract Derived {
    function test(address _addr) external {
        Base _b = new Base();
        assert(_b.x(0) != 0);
    }
}
