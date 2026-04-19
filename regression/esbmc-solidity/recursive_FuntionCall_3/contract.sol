// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract RecursiveFailWithDepth {
    uint public callCount = 0;

    function f(uint depth) internal {
        callCount += 1;

        if (depth > 0) {
            f(depth - 1);
        }

        assert(callCount < 5); 
    }

    function start() public {
        f(4); 
    }
}
