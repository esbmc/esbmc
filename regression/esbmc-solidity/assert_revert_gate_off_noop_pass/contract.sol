// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// Does NOT reference __ESBMC_reverted: the feature gate stays off and lowering
// is byte-for-byte unchanged (revert still prunes; assert verifies normally).
contract D {
    uint256 s;
    function f(uint256 x) public { require(x < 10, "g"); s = x; assert(s < 10); }
}
