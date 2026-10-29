// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for get_func_modifier: when a function with parameters is
// wrapped by a modifier, the synthetic `<func>_<mod>` aux function must
// inherit the wrapped function's parameters. Otherwise the inlined body's
// references to those parameters resolve to symbol names scoped to the
// aux function that were never declared, and symex aborts with
// `value_set: unknown symbol ...@<param>#<id>`.
//
// The assertion only references a modifier-wrapped function's own
// parameters via local variables, so the multi-transaction harness
// cannot perturb it.

contract C {
    modifier onlyPositive(uint256 gate) {
        require(gate > 0, "gate must be positive");
        _;
    }

    function addPair(uint256 x, uint256 y)
        external
        pure
        onlyPositive(x)
        returns (uint256)
    {
        return x + y;
    }

    function go() external view {
        uint256 r = this.addPair(1, 2);
        assert(r == 3);
    }
}
