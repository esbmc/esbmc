// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// build_tod_clone_helper assumes c->$address != base->$address.
// Verifies that two cloned instances have distinct on-chain addresses,
// so they don't collide in the EOA balance map or in the
// _ESBMC_get_obj address dispatch.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256 public x;
    function set(uint256 v) public { x = v; }
}

contract H {
    function check(uint256 v) public {
        C base = new C();
        base.set(v);
        C clone = __ESOL_deep_copy(base);
        assert(address(clone) != address(base));
        // Sanity: clone's value matches at the moment of cloning.
        assert(clone.x() == v);
    }
}
