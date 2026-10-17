// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: cross-contract single-level mapping read-after-write.
// `c.set(k, v)` writes `c.m[k] = v`; the harness then reads
// `c.m(k)` via the auto-generated public getter and must see v.
// (Was broken transiently when the public-getter path over-folded
// the single-key index — fixed by single-arg=no-fold.)
contract A {
    mapping(address => uint256) public m;
    function set(address k, uint v) public { m[k] = v; }
}

contract Test {
    function check(address k, uint v) public {
        A c = new A();
        c.set(k, v);
        assert(c.m(k) == v);
    }
}
