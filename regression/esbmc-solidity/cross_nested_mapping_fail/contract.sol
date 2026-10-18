// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Pair of cross_nested_mapping_pass: counter-claim must be refuted.
contract A {
    mapping(address => mapping(address => uint256)) public m;
    function set(address x, address y, uint v) public { m[x][y] = v; }
}

contract Test {
    function check(address x, address y, uint v) public {
        A c = new A();
        c.set(x, y, v);
        assert(c.m(x, y) == v + 1);
    }
}
