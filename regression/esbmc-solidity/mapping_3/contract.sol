// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base2 {
    mapping(string => int8) test2;

    constructor() {
        test2["1"] = 1;
        assert(test2["1"] != 0);
    }
}
