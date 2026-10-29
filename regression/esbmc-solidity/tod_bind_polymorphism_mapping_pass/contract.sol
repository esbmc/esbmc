// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: per-pointer polymorphism through `C(_addr)` cast.
//
// A1 and A2 are structurally identical, so ESBMC clusters them in the
// same structural-typing set.  `A1 alias = A1(address(c2))` creates an
// A1-typed pointer whose *runtime binding* must resolve to A2 because
// its address matches `_ESBMC_Object_A2.$address`.
//
// Pre-fix: `_ESBMC_bind_cname` was a shared field on the singleton
// struct — there was no per-pointer storage, so `alias.m(k)` routed
// statically to `_ESBMC_Object_A1`'s (empty) mapping and disagreed
// with `c2.m(k)` (the real write target).  VERIFICATION FAILED.
//
// Post-fix: each contract-typed local has a `$bind` shadow symbol
// written at cast time (address-match if-ladder) and at `new` time
// (declared type).  The mapping-getter polymorphism reads the shadow
// to choose the singleton — `alias.m(k)` and `c2.m(k)` both route to
// A2's storage and agree.  VERIFICATION SUCCESSFUL.
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
        A1 c1 = new A1();                // put A1 in newContractSet so
                                          // should_treat_as_new("A1") holds
        A2 c2 = new A2();
        c2.set(k, a);                    // writes _ESBMC_Object_A2.m[k] = a
        A1 alias_ = A1(address(c2));     // alias_'s $bind shadow = "A2"
        assert(alias_.m(k) == c2.m(k));  // both read A2's m[k] = a
        c1;                               // silence unused-local warning
    }
}
