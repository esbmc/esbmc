// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library SafeMath {
        /**
     * @dev Returns the multiplication of two unsigned integers, reverting on overflow.
     */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        // Gas optimization: this is cheaper than requiring 'a' not being zero,
        // but the benefit is lost if 'b' is also tested.
        // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
        if (a == 0) {
            return 0;
        }

        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }
}

contract MulExample {
    function safeMul(uint256 x, uint256 y) public pure returns (uint256) {
        return SafeMath.mul(x, y);
    }
}