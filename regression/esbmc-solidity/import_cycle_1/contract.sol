// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
import { IB } from "./contract2.sol";

library Util {
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }
}

contract A {
    using Util for uint256;
    IB public b;
    function f(uint256 x, uint256 y) external pure returns (uint256) {
        return x.max(y);
    }
}
