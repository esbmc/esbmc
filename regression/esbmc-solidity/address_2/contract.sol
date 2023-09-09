// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    address x = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;

    function comp() public {
        address payable addr3 = payable(x);
        assert(x == addr3);
    }
}
