// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// Real-contract shape: a mapping store with a require guard on the key.
contract Store {
    mapping(uint256 => uint256) m;
    function set(uint256 k, uint256 v) external { require(k != 0, "zero key"); m[k] = v; }
}
contract Harness {
    Store s;
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}
    constructor() { s = new Store(); }
    function check(uint256 k, uint256 v) public {
        __ESBMC_assume(k == 0);
        s.set(k, v);
        assert(__ESBMC_reverted());   // zero-key guard reverts
    }
}
