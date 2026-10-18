// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Mapping isolation property.  build_tod_clone_helper retargets each
// mapping field to the clone's fresh $address, so post-clone writes
// on the clone via mapping[k] live in a disjoint keyspace from base's
// mapping[k].  Verified property: writing through clone leaves base
// unchanged at the same key.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    mapping(uint256 => uint256) public m;
    function set(uint256 k, uint256 v) public { m[k] = v; }
    function get(uint256 k) public view returns (uint256) { return m[k]; }
}

contract H {
    function check(uint256 k, uint256 v_base, uint256 v_clone) public {
        if (v_base == v_clone) return;
        C base = new C();
        base.set(k, v_base);
        C clone = __ESOL_deep_copy(base);
        // write through clone only
        clone.set(k, v_clone);
        // base must still hold its original value at k
        assert(base.get(k) == v_base);
    }
}
