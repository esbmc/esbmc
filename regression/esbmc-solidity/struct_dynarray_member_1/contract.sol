// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0 <0.9.0;

// Struct with dynamic array member, accessed via storage ref.
// From Solidity docs "Arrays" section (ArrayContract.StructType).

contract ArrayContract {
    struct StructType {
        uint[] contents;
        uint moreInfo;
    }
    StructType s;

    function f() public {
        uint[] memory c = new uint[](2);
        c[0] = 10;
        c[1] = 20;
        StructType storage g = s;
        g.moreInfo = 2;
        g.contents = c;
        assert(s.moreInfo == 2);
    }
}
