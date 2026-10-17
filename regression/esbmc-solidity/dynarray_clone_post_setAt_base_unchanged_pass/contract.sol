// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S0 — clone-side setAt must not change base element.
// Real Solidity: clone.setAt(0, b) doesn't mutate base.arr[0].
// Today: alias propagates the write back to base.
// Will flip to CORE at Stage S3 (T1.1).
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function setAt(uint256 i, uint256 v) public { arr[i] = v; }
    function get(uint256 i) public view returns (uint256) { return arr[i]; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        if (a == b) return;
        C base = new C();
        base.push(a);
        C clone = __ESOL_deep_copy(base);
        clone.setAt(0, b);
        assert(base.get(0) == a);
    }
}
