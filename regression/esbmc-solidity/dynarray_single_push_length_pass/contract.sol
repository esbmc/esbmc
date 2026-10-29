// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S0 — single-instance push then read length.
// Real Solidity: c.arr.length == 1 after exactly one push.
// Today (heap-malloc model + memcpy unwind hazard): the assertion either
// vacuously passes via 0-VCC truncation at low --unwind, or fails outright
// because the malloc'd buffer's header isn't tracked through the global
// `arr` symbol re-assignment.  Will flip to CORE at Stage S1 (T1.1) once
// state-var dyn-arrays use the unified addr-keyed SMT model.
contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function len() public view returns (uint256) { return arr.length; }
}

contract H {
    function check(uint256 v) public {
        C c = new C();
        c.push(v);
        assert(c.len() == 1);
    }
}
