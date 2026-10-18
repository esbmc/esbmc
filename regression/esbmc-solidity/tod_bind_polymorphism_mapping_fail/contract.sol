// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Adversarial counterpart to tod_bind_polymorphism_mapping_pass.
//
// Same cast-polymorphism setup — A1 alias bound to A2's instance via
// `A1(address(c2))`, but the assertion now compares against `c1.m(k)`
// (A1's empty mapping).  With the per-pointer shadow routing:
//   - alias_.m(k) reads A2's storage → value a (from c2.set earlier)
//   - c1.m(k)     reads A1's storage → 0 (no write)
// Nondet `a` can differ from 0, so the assertion is refuted.  This
// test guards against the polymorphism fix accidentally collapsing
// both reads onto the same singleton (which would spuriously verify).
contract A1 {
    mapping(address => uint256) public m;
    function set(address k, uint v) public { m[k] = v; }
}
contract A2 {
    mapping(address => uint256) public m;
    function set(address k, uint v) public { m[k] = v; }
}

contract Test {
    function check(address k, uint a) public {
        A1 c1 = new A1();
        A2 c2 = new A2();
        c2.set(k, a);
        A1 alias_ = A1(address(c2));
        assert(alias_.m(k) == c1.m(k));   // must FAIL
    }
}
