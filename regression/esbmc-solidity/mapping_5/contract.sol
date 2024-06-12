// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    constructor() {
        test[-1] = -11;
        assert(test[-1] == 0);
    }
    mapping(int => int) test;
}
