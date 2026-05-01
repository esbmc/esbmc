// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Test: nested mapping store/load -- should FAIL
contract NestedMapFail {
    mapping(uint => mapping(uint => uint)) data;

    function test() public {
        data[1][2] = 100;
        assert(data[1][2] == 999);
    }
}
