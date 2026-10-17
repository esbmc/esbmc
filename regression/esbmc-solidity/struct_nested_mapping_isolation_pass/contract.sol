// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// B8 follow-up: a mapping field nested inside a user-struct state var
// must get its own per-instance addr so two C instances don't alias on
// `bx.m.addr=0`.  The Phase 2 ctor walker
// (emit_ctor_deep_init_fixup, solidity_convert_constructor.cpp) now
// emits `this->bx.m = {base=&_ESBMC_inf_C_bx_m[0], mid=N, addr=this->$address}`
// at the same level it emits calloc for struct-internal pointer-backed
// arrays.  Verified by writing through c1 and c2 at the same key and
// asserting c1.get(k) returns its own value.
contract C {
    struct Box { mapping(uint256 => uint256) m; }
    Box internal bx;
    function set(uint256 k, uint256 v) public { bx.m[k] = v; }
    function get(uint256 k) public view returns (uint256) { return bx.m[k]; }
}

contract H {
    function check(uint256 k, uint256 v1, uint256 v2) public {
        if (v1 == v2) return;
        C c1 = new C();
        C c2 = new C();
        c1.set(k, v1);
        c2.set(k, v2);
        assert(c1.get(k) == v1);
    }
}
