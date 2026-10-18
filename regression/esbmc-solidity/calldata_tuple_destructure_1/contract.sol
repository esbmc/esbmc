// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;
pragma abicoder v2;

contract C {
    function g(uint[][2] calldata s)
        internal
        pure
        returns (uint, uint[] calldata)
    {
        return (s[0][1], s[1]);
    }

    function f(uint, uint[][2] calldata s, uint)
        external
        pure
        returns (uint, uint)
    {
        (uint x, uint[] calldata y) = g(s);
        return (x, y[0]);
    }
}
