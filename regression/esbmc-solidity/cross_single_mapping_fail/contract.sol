// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pair of cross_single_mapping_pass: counter-claim must be refuted.
contract A {
    mapping(address => uint256) public m;
    function set(address k, uint v) public { m[k] = v; }
}

contract Test {
    function check(address k, uint v) public {
        A c = new A();
        c.set(k, v);
        assert(c.m(k) == v + 1);
    }
}
