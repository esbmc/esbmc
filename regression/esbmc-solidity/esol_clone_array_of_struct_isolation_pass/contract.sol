// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
// Array-of-struct-of-primitives isolation: post-clone write on base
// MUST NOT be visible via clone.  Since P has only scalar fields,
// needs_clone_deep_fixup is false and the walker uses single arrcpy
// (memcpy), which bit-copies all 3 structs into clone's fresh buffer.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    struct P { uint256 x; uint256 y; }
    P[3] public arr;
    function setXY(uint256 i, uint256 x, uint256 y) public {
        arr[i].x = x;
        arr[i].y = y;
    }
    function getX(uint256 i) public view returns (uint256) { return arr[i].x; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        require(a != b);
        C base = new C();
        base.setXY(2, a, 0);
        C clone = __ESOL_deep_copy(base);
        base.setXY(2, b, 0); // mutate base
        // clone still sees `a`, not `b`.
        assert(clone.getX(2) == a);
    }
}
