// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// String field round-trip through clone + keccak.  Historically this
// combination crashed ESBMC on symex due to a uint256/BytesStatic
// struct mismatch in the keccak256 hash_needs_nondet fallback; the
// fix (in solidity_convert_expr.cpp) now emits a typed nondet bytes32
// directly when the argument is a raw bytes struct.
function __ESOL_deep_copy(C src) pure returns (C) { return src; }

contract C {
    string public s;
    function set(string calldata _s) public { s = _s; }
    function digest() public view returns (bytes32) {
        return keccak256(bytes(s));
    }
}

contract H {
    function check(string calldata _s) public {
        C base = new C();
        base.set(_s);
        C clone = __ESOL_deep_copy(base);
        // Both instances run the same nondet abstraction on the hash;
        // since the abstraction is emitted at each call site, each call
        // is its own fresh nondet and we only assert the non-crashing
        // execution (not equality of unrelated nondets).
        bytes32 dc = clone.digest();
        bytes32 db = base.digest();
        assert(dc == dc);
        assert(db == db);
    }
}
