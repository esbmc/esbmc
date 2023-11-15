// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.4.0;

contract Base {
    function test_op() public {
        int x = 1;
        assert(!(~x-- != 65535));
    }
}
