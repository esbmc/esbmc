// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// Three sequential external calls in one harness: the flag reflects only the
// most recent call (cleared at each external boundary).
contract G { function step(uint256 x) external pure { require(x < 10, "g"); } }
contract Harness {
    G g;
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}
    constructor() { g = new G(); }
    function check(uint256 a, uint256 b, uint256 c) public {
        g.step(a);
        __ESBMC_assume(!__ESBMC_reverted());   // a < 10
        g.step(b);
        __ESBMC_assume(__ESBMC_reverted());    // b >= 10
        g.step(c);
        if (c < 10) assert(!__ESBMC_reverted());  // last call governs the flag
        else assert(__ESBMC_reverted());
    }
}
