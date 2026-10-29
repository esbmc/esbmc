// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// Pin the input range, then prove the call reverts for it.
contract Bank { function deposit(uint256 x) external pure { require(x < 1000, "cap"); } }
contract Harness {
    Bank bank;
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}
    constructor() { bank = new Bank(); }
    function check(uint256 x) public {
        __ESBMC_assume(x >= 1000);
        bank.deposit(x);
        assert(__ESBMC_reverted());   // proven: deposit reverts for x >= 1000
    }
}
