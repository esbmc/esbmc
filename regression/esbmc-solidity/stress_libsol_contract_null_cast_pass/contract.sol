// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for the `migrate expr failed` crash when a contract-typed
// null cast `ContractType(0)` reaches the goto layer as a constant_exprt
// of type `pointer→tag-Contract` with an empty value() slot. The fix
// replaces such literals with a proper gen_zero(pointer) before they
// reach migrate_expr.

contract Pool { uint256 public x; }

contract Factory {
    Pool public p;

    function check() public view {
        // require() compares a state-var Pool* against `Pool(0)`,
        // which used to crash goto-conversion with `migrate expr failed`.
        require(p == Pool(address(0)), "exists");
        // p is uninitialised state, so the require holds and we reach
        // a trivially true assertion.
        assert(p == Pool(address(0)));
    }
}
