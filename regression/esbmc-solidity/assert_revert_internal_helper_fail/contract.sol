// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
contract C {
    function __ESBMC_reverted() internal returns (bool) {}
    function inner2(uint256 x) internal pure { require(x < 10, "deep"); }
    function inner1(uint256 x) internal pure { inner2(x); }
    function check(uint256 x) public {
        inner1(x);
        assert(__ESBMC_reverted());   // FALSE for x < 10 (no revert) -> FAILED
    }
}
