// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.17;

contract Base {
    function Conv() public {
        uint32 a = 0x432178;
        uint16 b = uint16(a); // b will be 0x2178 now
        assert(b == 0x2178);

        uint16 aa = 0x4356;
        uint32 bb = uint32(aa); // bb will be 0x00004356 now
        assert(bb == 0x00004356);
    }
}
