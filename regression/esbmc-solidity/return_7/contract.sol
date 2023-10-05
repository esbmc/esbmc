// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Base {
    constructor() {}

    function sum(uint k) public returns (uint) {
        if (k > 0) {
            return k + sum(k - 1);
        } else {
            for (int i = 10; i > 0; i--) for (int j = 10; j > 0; j--) k--;
            assert(k == 0);
            return 0;
        }
    }

    function main() public {
        sum(10);
    }
}
