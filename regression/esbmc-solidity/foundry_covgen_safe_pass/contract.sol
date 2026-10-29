// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

// --generate-foundry-testcase must not perturb a normal verification: with no
// counterexample there is nothing to reconstruct and the run stays SUCCESSFUL.
contract Safe {
    uint256 public x;
    function store(uint256 v) public {
        x = v;
        assert(x == v);
    }
}
