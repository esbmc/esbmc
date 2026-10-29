// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Cloning a derived contract that inherits a mapping field from a
// base.  Walks two stages of B8 plumbing:
//   1. Inherited mapping per-instance addr init in D's ctor (the
//      gate change in solidity_convert_decl.cpp:move_to_initializer)
//      so base.m.addr = base.$address is established before clone.
//   2. The Phase 1 clone walker (emit_clone_deep_copy_fixup) then
//      retargets clone.m.addr = clone.$address, giving the clone
//      its own keyspace.
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
        assert(base.get(k) == v_base);
    }
}
