// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Cloning a contract that nests a mapping inside a struct field.
// Walks two stages of the B8 plumbing:
//   1. Phase 2 ctor walker (emit_ctor_deep_init_fixup) initialises
//      base.bx.m.addr = base.$address before clone, so the base-side
//      mapping has a well-formed keyspace.
//   2. Phase 1 clone walker (emit_clone_deep_copy_fixup) recurses
//      into bx and retargets clone.bx.m.addr = clone.$address — this
//      branch was already correct (no clone-side change in this fix).
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
        assert(base.get(k) == v_base);
    }
}
