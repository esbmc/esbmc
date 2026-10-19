// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract C {
    function f(uint256 a) public pure returns (uint256) {
        require(a == 2**200);
        uint256 r = a ** 1;
        assert(r == 2**200);
        return r;
    }
}
