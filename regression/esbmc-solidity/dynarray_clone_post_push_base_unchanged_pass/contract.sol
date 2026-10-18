// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S0 — clone-side push must not change base length.
// Real Solidity: `__ESOL_deep_copy(base)` produces an isolated clone;
// post-clone clone.push(v) leaves base.arr.length at its pre-clone value.
// Today: alias means clone.push reaches into base's keyspace.
// Will flip to CORE at Stage S3 (T1.1).
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function len() public view returns (uint256) { return arr.length; }
}

contract H {
    function check(uint256 a, uint256 b) public {
        if (a == b) return;
        C base = new C();
        base.push(a);
        C clone = __ESOL_deep_copy(base);
        clone.push(b);
        assert(base.len() == 1);
    }
}
