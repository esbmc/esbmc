// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

error Base1(int Base);
contract Base {
    int Base = 0;
    function test() public view {
        int Base = 1;
        if (Base == 0) {
            revert Base1(Base);
        }
        assert(Base == 1);
    }
}
