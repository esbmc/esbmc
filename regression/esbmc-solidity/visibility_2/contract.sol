// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;
contract Base {
    int8 public x = 2;
    function div_zero() private view
    {
        int y = 10/(x-2);
    }
}
