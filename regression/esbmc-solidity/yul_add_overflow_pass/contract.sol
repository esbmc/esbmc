// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Clean Yul `add` with --overflow-check: 5 + 10 cannot overflow,
    // so ESBMC's overflow check does NOT fire and verification passes.
    function check() public pure {
        uint256 a = 5;
        uint256 b = 10;
        uint256 r;
        assembly {
            r := add(a, b)
        }
        // Pin the precise sum for good measure.
        assert(r == 15);
    }
}
