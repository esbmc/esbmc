// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: cross-contract nested mapping read-after-write.
// `c.set(x, y, v)` writes via the combined-key path inside A; the
// harness reads `c.m(x, y)` via the auto-generated public getter
// using the SAME combined key on the SAME mapping_t entry.  The
// pointer-arithmetic alternative (`*(map_generic_get(...) + ...)`)
// does NOT work across the function-call boundary in ESBMC's
// pointer model — combined-key + map_uint_get is what makes this
// pair line up.
contract A {
    mapping(address => mapping(address => uint256)) public m;
    function set(address x, address y, uint v) public { m[x][y] = v; }
}

contract Test {
    function check(address x, address y, uint v) public {
        A c = new A();
        c.set(x, y, v);
        assert(c.m(x, y) == v);
    }
}
