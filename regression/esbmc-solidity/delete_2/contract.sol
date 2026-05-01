// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract DeleteTest {
    uint256 public value;

    function test() public {
        value = 100;
        delete value;
        // value is now 0, not 100
        assert(value == 100);  // should FAIL
    }
}
