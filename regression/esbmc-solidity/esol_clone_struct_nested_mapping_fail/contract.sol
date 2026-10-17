// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative dual of esol_clone_struct_nested_mapping_pass.  Asserts the
// OPPOSITE — that base.bx.m sees clone's post-clone write — which is
// what would happen if either the Phase 2 ctor init (this fix) or the
// Phase 1 clone walker's nested-mapping addr retargeting regressed.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    struct Box { mapping(uint256 => uint256) m; }
    Box internal bx;
    function set(uint256 k, uint256 v) public { bx.m[k] = v; }
    function get(uint256 k) public view returns (uint256) { return bx.m[k]; }
}

contract H {
    function check(uint256 k, uint256 v_base, uint256 v_clone) public {
        if (v_base == v_clone) return;
        C base = new C();
        base.set(k, v_base);
        C clone = __ESOL_deep_copy(base);
        clone.set(k, v_clone);
        // isolation => this FAILS
        assert(base.get(k) == v_clone);
    }
}
