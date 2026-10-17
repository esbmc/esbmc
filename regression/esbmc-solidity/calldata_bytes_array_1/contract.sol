// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

contract C {
    function f1(bytes[1] calldata a)
        external
        pure
        returns (uint256, uint256, uint256, uint256)
    {
        bytes memory b = a[0];
        require(b.length > 2);
        return (b.length, uint8(b[0]), uint8(b[1]), uint8(b[2]));
    }

    function g2(bytes[] calldata a) external pure returns (uint256[8] memory) {
        require(a.length > 1);
        bytes memory b0 = a[0];
        bytes memory b1 = a[1];
        require(b0.length > 1 && b1.length > 2);
        return [
            a.length,
            b0.length,
            uint8(b0[0]),
            uint8(b0[1]),
            b1.length,
            uint8(b1[0]),
            uint8(b1[1]),
            uint8(b1[2])
        ];
    }
}
