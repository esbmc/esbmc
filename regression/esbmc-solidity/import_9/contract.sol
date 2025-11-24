// SPDX-License-Identifier: GPL-3.0
import "./contract2.sol";
pragma solidity >=0.8.0;

contract Base {
    address x = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;
    T y = T(x);
    function test() public {
        y.test();
    }
}