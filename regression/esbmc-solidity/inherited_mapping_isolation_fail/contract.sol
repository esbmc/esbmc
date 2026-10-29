// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Negative dual of inherited_mapping_isolation_pass.  Asserts the
// OPPOSITE — that d1's read at k returns d2's write — which would
// only happen if the per-instance addr init regressed back to the
// pre-B8 alias-everywhere state.  If anyone reverts that gate change,
// this fail test flips to pass and the paired pass test flips to fail.
contract Base {
    mapping(uint256 => uint256) internal m;
}

contract D is Base {
    function set(uint256 k, uint256 v) public { m[k] = v; }
    function get(uint256 k) public view returns (uint256) { return m[k]; }
}

contract H {
    function check(uint256 k, uint256 v1, uint256 v2) public {
        if (v1 == v2) return;
        D d1 = new D();
        D d2 = new D();
        d1.set(k, v1);
        d2.set(k, v2);
        // isolation => this FAILS
        assert(d1.get(k) == v2);
    }
}
