// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Probes whether `address(this).balance` reads from the same SSA cell on
// repeated access (i.e. `this->$balance` for a known instance) instead of
// returning a fresh nondet each time via get_aux_property_function.
//
// Before the SMTChecker-style balance model fix, both reads returned
// independent nondet uint256 values, so `a == b` could fail trivially in
// unbound mode.  After the fix, they hit the same member access and the
// equality holds.
contract Bal {
    constructor() payable {}

    function probe() public view {
        uint a = address(this).balance;
        uint b = address(this).balance;
        assert(a == b);
    }
}
