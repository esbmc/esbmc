// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S0 — single-instance push then read element [0].
// Real Solidity: c.arr[0] == v after c.push(v).
// Today: heap-malloc model + memcpy hazard. Will flip to CORE at S2 (T1.1).
contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function get(uint256 i) public view returns (uint256) { return arr[i]; }
}

contract H {
    function check(uint256 v) public {
        C c = new C();
        c.push(v);
        assert(c.get(0) == v);
    }
}
