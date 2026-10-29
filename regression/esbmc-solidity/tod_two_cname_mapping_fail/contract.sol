// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: TOD-style harness with two structurally identical
// contracts (A1 and A2) that both contain a nested mapping.  When
// cname_set.size() > 1 (structurally typed cluster), cross-contract
// function calls dispatch through `_ESBMC_Object_<cname>` — writes
// land in the singleton's storage.  The public-mapping-getter read
// path must route through the same singleton (not the local pointer)
// or both reads return 0/nondet and the harness can't tell c1 from c2.
//
// Sequence: c1 writes a then b (ends at b); c2 writes b then a
// (ends at a).  assert(c1.m(k,k) == c2.m(k,k)) is refuted whenever
// a != b — but only if reads see the writes.
contract A1 {
    mapping(address => mapping(address => uint256)) public m;
    function set(address k, uint v) public { m[k][k] = v; }
}
contract A2 {
    mapping(address => mapping(address => uint256)) public m;
    function set(address k, uint v) public { m[k][k] = v; }
}

contract Test {
    function check(address k, uint a, uint b) public {
        A1 c1 = new A1();
        A2 c2 = new A2();
        c1.set(k, a); c1.set(k, b);
        c2.set(k, b); c2.set(k, a);
        assert(c1.m(k, k) == c2.m(k, k));
    }
}
