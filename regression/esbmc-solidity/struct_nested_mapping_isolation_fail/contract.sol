// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative dual of struct_nested_mapping_isolation_pass.  Asserts the
// OPPOSITE — that c1 sees c2's write — which would only happen if the
// per-instance ctor init for nested mapping fields regressed.  If
// anyone reverts the Phase 2 walker mapping-leaf case, this fail test
// flips to pass and the paired pass test flips to fail.
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
        // isolation => this FAILS
        assert(c1.get(k) == v2);
    }
}
