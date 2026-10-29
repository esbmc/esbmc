// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract H {
    // Yul `add` is 2^256 wrap (no checked-arithmetic revert). With
    // --overflow-check enabled, ESBMC's automatic overflow detector
    // fires on `add(MAX, 1)` and reports "arithmetic overflow on add"
    // — no manual assert needed. This is the bug the user actually
    // wants surfaced when porting Solidity 0.8+ code into assembly.
    function check() public pure {
        uint256 a = type(uint256).max;
        uint256 b = 1;
        uint256 r;
        assembly {
            r := add(a, b)
        }
        // No assert — the overflow check itself produces the violation.
    }
}
