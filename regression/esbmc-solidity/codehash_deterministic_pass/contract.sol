// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Per-address codehash determinism. Pre-fix, the fall-through branch
// in get_aux_property_function returned a fresh nondet_uint256 on
// every read, so `addr.codehash == addr.codehash` could fail (two
// independent nondets). Post-fix, `_ESBMC_codehash_of(addr)` looks up
// the address in the EOA pool and returns the slot's stored value, so
// repeated reads agree within a path.
contract C {
    function check(address a) public view {
        bytes32 h1 = a.codehash;
        bytes32 h2 = a.codehash;
        assert(h1 == h2);
    }
}
