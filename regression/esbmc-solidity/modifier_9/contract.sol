// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract D {
    uint x;
    uint y;

    modifier check() {
        _;
        assert(0 == 1);
    }

    function func2() public check returns (uint) {
        x = 1;
        return 0;
    }
}
