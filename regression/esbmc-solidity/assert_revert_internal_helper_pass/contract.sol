// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// Revert buried two levels deep in internal helpers is captured at the
// external boundary (mark + return relaxed lowering).
contract C {
    function __ESBMC_reverted() internal returns (bool) {}
    function inner2(uint256 x) internal pure { require(x < 10, "deep"); }
    function inner1(uint256 x) internal pure { inner2(x); }
    function check(uint256 x) public {
        inner1(x);
        if (x >= 10) assert(__ESBMC_reverted());   // deep revert observed
        else assert(!__ESBMC_reverted());          // cleared at entry, no revert
    }
}
