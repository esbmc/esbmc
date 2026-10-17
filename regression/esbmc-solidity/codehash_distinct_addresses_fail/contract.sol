// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Fail-dual: two distinct addresses get independent nondet codehashes,
// so asserting equality must fail.  This pins the over-approximation
// in the right direction — codehashes are NOT correlated across
// addresses.
contract C {
    function check(address a, address b) public view {
        if (a == b) return;
        bytes32 h1 = a.codehash;
        bytes32 h2 = b.codehash;
        assert(h1 == h2);
    }
}
