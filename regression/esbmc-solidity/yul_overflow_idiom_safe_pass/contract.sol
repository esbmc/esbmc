// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Demonstrates the SAFE idiom: detect Yul-add overflow inline using
    // `lt(r, a)`. After `r := add(a, b)`, r < a iff the add wrapped
    // (a + b cannot shrink for non-negative b in checked arithmetic).
    // ESBMC must precisely recognise both the wrap AND the detection.
    function check() public pure {
        uint256 a = type(uint256).max;
        uint256 b = 1;
        uint256 r;
        uint256 ovf;
        assembly {
            r := add(a, b)
            ovf := lt(r, a)   // 1 iff overflow occurred
        }
        assert(ovf == 1);
        assert(r == 0);
    }
}
