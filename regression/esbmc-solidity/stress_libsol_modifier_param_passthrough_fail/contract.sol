// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual to pass: same shape, deliberately-wrong assertion so the run must
// actually reach the assert via the modifier-wrapped call path.

contract C {
    modifier onlyPositive(uint256 gate) {
        require(gate > 0, "gate must be positive");
        _;
    }

    function addPair(uint256 x, uint256 y)
        external
        pure
        onlyPositive(x)
        returns (uint256)
    {
        return x + y;
    }

    function go() external view {
        uint256 r = this.addPair(1, 2);
        assert(r == 99); // wrong
    }
}
