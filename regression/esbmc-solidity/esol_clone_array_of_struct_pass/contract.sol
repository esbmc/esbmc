// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// Array-of-struct-of-primitives: post-clone equality.  `P` has no
// pointer-backed fields (no mapping, no nested array), so
// needs_clone_deep_fixup returns false and the walker routes through
// a single _ESBMC_arrcpy that memcpy's all N structs in one shot.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    struct P { uint256 x; uint256 y; }
    P[3] public arr;
    function setXY(uint256 i, uint256 x, uint256 y) public {
        arr[i].x = x;
        arr[i].y = y;
    }
    function getX(uint256 i) public view returns (uint256) { return arr[i].x; }
    function getY(uint256 i) public view returns (uint256) { return arr[i].y; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        C base = new C();
        base.setXY(1, a, b);
        C clone = __ESOL_deep_copy(base);
        // Read-after-clone: see the values base wrote pre-clone.
        assert(clone.getX(1) == a);
        assert(clone.getY(1) == b);
    }
}
