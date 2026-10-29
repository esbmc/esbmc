// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

// new T[N][](size) — allocating a dynamic array of fixed-size sub-arrays.
// From Solidity docs "Arrays" section (ArrayContract.createMemoryArray).

contract ArrayContract {
    function test_new_fixdyn() public pure {
        uint[2][] memory arrayOfPairs = new uint[2][](3);
        assert(arrayOfPairs.length == 3);

        bool[2][] memory flags = new bool[2][](2);
        assert(flags.length == 2);
    }
}
