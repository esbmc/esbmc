// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// Headline use case: verify "whenever A does not revert, B reverts".
contract A { function test(uint256 x) external pure { require(x < 100, "A big"); } }
contract B { function test(uint256 x) external pure { require(x >= 100, "B small"); } }
contract Harness {
    A a; B b;
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}
    constructor() { a = new A(); b = new B(); }
    function check(uint256 x) public {
        a.test(x);
        __ESBMC_assume(!__ESBMC_reverted());  // A did not revert => x < 100
        b.test(x);
        assert(__ESBMC_reverted());           // => B must revert
    }
}
