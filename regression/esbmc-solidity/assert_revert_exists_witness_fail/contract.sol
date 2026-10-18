// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// Existence witness: FAILED counterexample is a path where A does NOT revert but B DOES.
contract A { function test(uint256 x) external pure { require(x < 100, "A big"); } }
contract B { function test(uint256 x) external pure { require(x >= 100, "B small"); } }
contract Harness {
    A a; B b;
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}
    constructor() { a = new A(); b = new B(); }
    function check(uint256 x) public {
        a.test(x);
        __ESBMC_assume(!__ESBMC_reverted());  // A did not revert
        b.test(x);
        assert(!__ESBMC_reverted());          // claim B never reverts -> FAILS, witness x<100
    }
}
