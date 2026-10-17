// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

// Covering both branches of `probe` requires TWO different constructions
// (cap > 100 vs cap <= 100). The generated Foundry test must split into two
// test contracts, each with its own setUp() deploying a distinct instance,
// AND must actually CALL probe() -- whose own argument is irrelevant to the
// branch and is defaulted. Regression guard for the transaction-sequence
// reconstruction + per-construction setUp split (empty bodies before the fix).
contract Vault {
    uint256 public cap;
    constructor(uint256 c) { cap = c; }
    function probe(uint256 x) public view returns (uint256) {
        if (cap > 100) return x;
        return x + 1;
    }
}
