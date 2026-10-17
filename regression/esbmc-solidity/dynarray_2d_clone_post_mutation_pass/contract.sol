// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S4 — 2D clone post-mutation isolation.
// After clone, base.setAt must not be visible through clone.
// Today: outer-fold collision lets base's write reach clone's slot.
// Will flip to CORE at Stage S5 (T1.1).
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[][] public arr;
    function pushNew(uint256 size) public { arr.push(new uint256[](size)); }
    function setAt(uint256 i, uint256 j, uint256 v) public { arr[i][j] = v; }
    function getAt(uint256 i, uint256 j) public view returns (uint256) { return arr[i][j]; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        if (a == b) return;
        C base = new C();
        base.pushNew(2);
        base.setAt(0, 0, a);
        C clone = __ESOL_deep_copy(base);
        base.setAt(0, 0, b);
        // clone[0][0] must remain `a` (its pre-clone value).
        assert(clone.getAt(0, 0) == a);
    }
}
