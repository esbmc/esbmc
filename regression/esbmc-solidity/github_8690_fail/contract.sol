// SPDX-License-Identifier: MIT
// #8690: a shift count is not truncated to the left operand's type.
pragma solidity ^0.8.0;
contract C {
    function shifts(uint256 n) public pure {
        require(n == 256 || n == 2**32 || n == 2**255);
        uint8 a = 128;
        int8 s = -128;
        uint32 w = 1;
        assert(a >> n != 0);
        assert(s >> n != -1);
        assert(w << n != 0);
        uint16 c = 1;
        c <<= n;
        assert(c != 0);
        s >>= n;
        assert(s != -1);
    }
    function small(uint256 n) public pure {
        require(n == 3);
        uint8 a = 1;
        int16 s = -128;
        assert(a << n != 8);
        assert(s >> n != -16);
    }
    function bytesShift(uint256 n) public pure {
        require(n == 8 || n == 2**32 + 8);
        bytes4 b = 0x12345678;
        assert(n != 8 || b << n != bytes4(0x34567800));
        assert(n == 8 || b << n != bytes4(0));
        assert(n == 8 || b >> n != bytes4(0));
    }
}
