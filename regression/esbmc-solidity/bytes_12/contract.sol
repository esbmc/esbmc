// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract Base {

    function test() public {
        bytes memory x;
        assert(x[1] == 0);
    }
}
