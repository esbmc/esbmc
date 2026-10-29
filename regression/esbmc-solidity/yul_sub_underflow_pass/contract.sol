// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Clean Yul `sub` with --overflow-check: 10 - 5 cannot underflow,
    // so ESBMC's overflow check does NOT fire.
    function check() public pure {
        uint256 a = 10;
        uint256 b = 5;
        uint256 r;
        assembly {
            r := sub(a, b)
        }
        assert(r == 5);
    }
}
