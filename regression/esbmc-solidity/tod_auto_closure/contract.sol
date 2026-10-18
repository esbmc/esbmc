// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// outerA / outerB write `x` only via internal helpers.  The candidate
// finder MUST follow the intra-contract call graph to discover the
// shared write target; targeted assertions must do the same to emit a
// useful equality check.  If either side regresses to body-only
// analysis, this test reports a vacuous SUCCESS instead of FAILED.
contract Closure {
    uint public x;

    constructor() { x = 0; }

    function _doubleX() internal { x = x * 2; }
    function _addOne() internal { x = x + 1; }

    function outerA() public { _doubleX(); }
    function outerB() public { _addOne(); }
}
