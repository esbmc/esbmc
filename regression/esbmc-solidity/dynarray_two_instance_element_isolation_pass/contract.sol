// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S0 — two-instance element isolation, NO clone.
// Real Solidity: c1's push doesn't reach c2's element [0]. Sequenced
// so that c2 pushes 0 FIRST, then c1 pushes v; the order ensures the
// shared-SMT-array alias would be c1's `v` (last write wins), so a
// passing read of `c2.get(0) == 0` requires per-instance element
// keyspace.
// Today: shared global `arr` makes c2.get(0) read c1's last-written v.
// Will flip to CORE at Stage S2 (T1.1).
contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function get(uint256 i) public view returns (uint256) { return arr[i]; }
}

contract H {
    function check(uint256 v) public {
        if (v == 0) return;  // make the FAIL distinguishable from default 0
        C c1 = new C();
        C c2 = new C();
        c2.push(0);  // first: c2's keyspace gets 0 at index 0
        c1.push(v);  // second: c1's keyspace gets v at index 0
        // Under alias: shared arr[0]=v (c1 wrote last), c2.get(0)=v ≠ 0. FAIL.
        // Under isolation: c2's keyspace[0]=0. PASS.
        assert(c2.get(0) == 0);
    }
}
