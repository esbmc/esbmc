// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// Revert from a modifier guard (onlyOwner / whenNotPaused family) is captured.
contract Vault {
    uint256 public total;
    modifier capped(uint256 v) { require(v <= 100, "cap exceeded"); _; }
    function deposit(uint256 v) external capped(v) { total += v; }
}
contract Harness {
    Vault v;
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}
    constructor() { v = new Vault(); }
    function check(uint256 amt) public {
        __ESBMC_assume(amt > 100);
        v.deposit(amt);
        assert(__ESBMC_reverted());   // modifier guard revert observed
    }
}
