// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract Base {
    bytes1 y;
    constructor() {
        assert(y[0] == 0);
    }
    function test() public {
        bytes32 x;
        assert(x[1] == 0);
    }
}
