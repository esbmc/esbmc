// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.8;

type UFixed256x18 is uint256;

uint constant multiplier2 = 10 ** 18;

library FixedMath {
    uint constant public multiplier = 10 ** 18;

    function add(
        UFixed256x18 a,
        UFixed256x18 b
    ) internal pure returns (UFixed256x18) {
        return UFixed256x18.wrap(UFixed256x18.unwrap(a) + UFixed256x18.unwrap(b));
    }

    function mul(
        UFixed256x18 a,
        uint256 b
    ) internal pure returns (UFixed256x18) {
        return UFixed256x18.wrap(UFixed256x18.unwrap(a) * b);
    }

    function floor(UFixed256x18 a) internal pure returns (uint256) {
        return UFixed256x18.unwrap(a) / multiplier;
    }

    function toUFixed256x18(uint256 a) internal pure returns (UFixed256x18) {
        return UFixed256x18.wrap(a * multiplier);
    }
}

contract FixedMathTest {
    using FixedMath for UFixed256x18;

    UFixed256x18 public valueA;
    UFixed256x18 public valueB;
    UFixed256x18 public result;

    constructor() {
        valueA = FixedMath.toUFixed256x18(5);
        valueB = FixedMath.toUFixed256x18(3);
    }

    function testAdd() public {
        result = valueA.add(valueB);
        uint256 intPart = result.floor();
        assert(intPart == 8);
    }

    function testMul() public {
        result = valueA.mul(2);
        uint256 intPart = result.floor();
        assert(intPart == 10);
    }

    function testFloor() public {
        uint256 intPart = valueB.floor();
        assert(intPart == 3);
    }
}
