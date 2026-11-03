// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// #8708: Solidity truncates a shift, so << carries no overflow check.
contract C {
    function f(uint256 a) public pure {
        uint256 b = a << 255;
        assert(b == 0 || b == 1 << 255);
    }
}
