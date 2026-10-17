// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// `revert CustomError(args)` is captured by the flag (args dropped).
contract Token {
    error TooBig(uint256 got, uint256 cap);
    function mint(uint256 x) external pure { if (x > 100) revert TooBig(x, 100); }
}
contract Harness {
    Token t;
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}
    constructor() { t = new Token(); }
    function check(uint256 x) public {
        __ESBMC_assume(x > 100);
        t.mint(x);
        assert(__ESBMC_reverted());   // custom-error revert observed
    }
}
