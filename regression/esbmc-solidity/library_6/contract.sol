// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library SafeMath {
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "SafeMath: division by zero");
        return a / b;
    }
}

contract DivExample {
    function safeDiv(uint256 x, uint256 y) public pure returns (uint256) {
        return SafeMath.div(x, y);
    }
}