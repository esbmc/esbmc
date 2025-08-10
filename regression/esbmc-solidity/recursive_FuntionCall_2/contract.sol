// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract NestedAssertionFail {
    uint public counter = 0;

    function outer() public {
        counter += 1;
        assert(counter < 5);  
        outer();
    }

    function inner() internal {
        counter += 1;
        outer();
    }
}
