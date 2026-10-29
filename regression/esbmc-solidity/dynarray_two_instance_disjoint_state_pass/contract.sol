// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S0 — two-instance disjoint state, NO clone.
// Real Solidity: pushing distinct values into two instances produces
// distinct element [0] reads.
// Today: alias makes both reads return the same (last-written) value.
// Will flip to CORE at Stage S2 (T1.1).
contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function get(uint256 i) public view returns (uint256) { return arr[i]; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        if (a == b) return;
        C c1 = new C();
        C c2 = new C();
        c1.push(a);
        c2.push(b);
        assert(c1.get(0) == a);
        assert(c2.get(0) == b);
    }
}
