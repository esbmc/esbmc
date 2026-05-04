// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract DeleteTest {
    uint256 public value;
    bool public flag;
    uint8 public small;

    function test() public {
        value = 42;
        flag = true;
        small = 255;

        // delete resets to default values
        delete value;
        delete flag;
        delete small;

        assert(value == 0);
        assert(flag == false);
        assert(small == 0);
    }
}
