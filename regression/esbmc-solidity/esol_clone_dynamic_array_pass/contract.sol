// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Dynamic array post-clone equality.  After cloning, the clone must
// observe the same length and the same element values that base set.
// Whether the clone deep-copies the underlying buffer or shares a
// pointer, the read-after-write equality must hold.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    uint256[] public arr;
    function push(uint256 v) public { arr.push(v); }
    function len() public view returns (uint256) { return arr.length; }
    function get(uint256 i) public view returns (uint256) { return arr[i]; }
}

contract H {
    function check(uint256 v0, uint256 v1) public {
        C base = new C();
        base.push(v0);
        base.push(v1);
        C clone = __ESOL_deep_copy(base);
        assert(clone.len() == 2);
        assert(clone.get(0) == v0);
        assert(clone.get(1) == v1);
    }
}
