// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Adversarial stress test for __ESOL_deep_copy on primitives.
// All four primitive fields (uint, address, bool, bytes32) must be
// bit-equal between base and clone immediately after the copy.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256 public u;
    address public a;
    bool public b;
    bytes32 public bz;

    function setAll(uint256 _u, address _a, bool _b, bytes32 _bz) public {
        u = _u; a = _a; b = _b; bz = _bz;
    }
}

contract H {
    function check(uint256 _u, address _a, bool _b, bytes32 _bz) public {
        C base = new C();
        base.setAll(_u, _a, _b, _bz);
        C clone = __ESOL_deep_copy(base);
        assert(clone.u() == base.u());
        assert(clone.a() == base.a());
        assert(clone.b() == base.b());
        assert(clone.bz() == base.bz());
    }
}
