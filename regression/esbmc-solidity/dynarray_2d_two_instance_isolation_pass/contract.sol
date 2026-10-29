// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S4 — 2D state-var dyn-array two-instance element isolation.
// Real Solidity: c1's write to arr[0][0] does not reach c2.
// Today: S2's XOR-fold `_ESBMC_dynarr_idx(addr, 0)` is collision-vulnerable
// against adversarial nondet addresses — the SMT solver picks c1.addr and
// c2.addr such that fold(c1.addr, 0) == fold(c2.addr, 0), making both
// instances index the SAME inner row, so c2.get(0, 0) returns c1's `v`.
// Will flip to CORE at Stage S5 (T1.1) — addr-prefixed nested SMT arrays.
contract C {
    uint256[][] public arr;
    function pushNew(uint256 size) public { arr.push(new uint256[](size)); }
    function setAt(uint256 i, uint256 j, uint256 v) public { arr[i][j] = v; }
    function getAt(uint256 i, uint256 j) public view returns (uint256) { return arr[i][j]; }
}

contract H {
    function check(uint256 v) public {
        if (v == 0) return;
        C c1 = new C();
        C c2 = new C();
        c1.pushNew(2);  // c1.arr.length = 1, inner row [0,0]
        c2.pushNew(2);  // c2.arr.length = 1, inner row [0,0]
        c1.setAt(0, 0, v);
        // c2 never wrote element [0][0]; must remain 0.
        assert(c2.getAt(0, 0) == 0);
    }
}
