// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T1.1 Stage S0 — two-instance length isolation, NO clone.
// Real Solidity: two `new C()` instances have independent storage; a push
// on c1 must not change c2.arr.length.
// Today: state-var dyn-array `arr` is a static-lifetime global symbol shared
// by all instances of the same contract type — c1.push(v) makes c2 see len 1.
// Will flip to CORE at Stage S1 (T1.1).
contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function len() public view returns (uint256) { return arr.length; }
}

contract H {
    function check(uint256 v) public {
        C c1 = new C();
        C c2 = new C();
        c1.push(v);
        assert(c2.len() == 0);
    }
}
