// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Clean Yul `mul` with --overflow-check: 2 * 3 cannot overflow.
    function check() public pure {
        uint256 a = 2;
        uint256 b = 3;
        uint256 r;
        assembly {
            r := mul(a, b)
        }
        assert(r == 6);
    }
}
