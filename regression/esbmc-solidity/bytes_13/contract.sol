// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract Base {
    bytes y;
    constructor() {
        assert(y[0] == 0);
    }

}
