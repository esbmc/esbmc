// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// A call that never reverts: __ESBMC_reverted() must be false afterwards.
contract Safe { function f(uint256 x) external pure returns (uint256) { return x; } }
contract Harness {
    Safe sf;
    function __ESBMC_reverted() internal returns (bool) {}
    constructor() { sf = new Safe(); }
    function check(uint256 x) public {
        sf.f(x);
        assert(!__ESBMC_reverted());   // no revert => flag stays cleared
    }
}
