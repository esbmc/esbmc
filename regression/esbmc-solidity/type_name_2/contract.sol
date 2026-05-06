// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract MyContract {
    uint256 public value;

    function test() public {
        // type(MyContract).name is accessible
        string memory n = type(MyContract).name;
        // Use a separate assertion that is falsifiable
        value = 42;
        delete value;
        assert(value == 100);  // should FAIL: value is 0
    }
}
