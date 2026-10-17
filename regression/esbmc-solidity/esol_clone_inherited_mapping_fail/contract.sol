// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative dual of esol_clone_inherited_mapping_pass.  Asserts the
// OPPOSITE — that base.m[k] sees the clone's post-clone write — which
// is what would happen if either the inherited-mapping ctor init or
// the clone walker's addr-retarget regressed.
function __ESOL_deep_copy(D src) pure returns (D) { return src; }

contract Base {
    mapping(uint256 => uint256) internal m;
}

contract D is Base {
    function set(uint256 k, uint256 v) public { m[k] = v; }
    function get(uint256 k) public view returns (uint256) { return m[k]; }
}

contract H {
    function check(uint256 k, uint256 v_base, uint256 v_clone) public {
        if (v_base == v_clone) return;
        D base = new D();
        base.set(k, v_base);
        D clone = __ESOL_deep_copy(base);
        clone.set(k, v_clone);
        // isolation => this FAILS
        assert(base.get(k) == v_clone);
    }
}
