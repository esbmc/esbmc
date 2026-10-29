// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Struct of primitives — every field must round-trip through the clone.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    struct Pair { uint256 x; address y; bool z; }
    Pair public p;
    function set(uint256 _x, address _y, bool _z) public {
        p.x = _x; p.y = _y; p.z = _z;
    }
}

contract H {
    function check(uint256 _x, address _y, bool _z) public {
        C base = new C();
        base.set(_x, _y, _z);
        C clone = __ESOL_deep_copy(base);
        (uint256 cx, address cy, bool cz) = clone.p();
        (uint256 bx, address by, bool bz) = base.p();
        assert(cx == bx);
        assert(cy == by);
        assert(cz == bz);
    }
}
