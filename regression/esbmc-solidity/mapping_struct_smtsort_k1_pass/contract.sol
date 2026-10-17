// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// CORE control / regression guard for mapping_struct_smtsort_k2_knownbug.
//
// K=1 single-level mapping whose value is a struct = array<Struct,inf>.
// This works today via the array_conv-with-struct-subtype route (the
// node flattener stores struct-sorted elements opaquely; tuple_node
// decomposes fields at AST level), so it is VERIFICATION SUCCESSFUL.
//
// It pins the working boundary: the Stage 2C fix MUST keep this
// byte-identical (K=1-byte-identical invariant, design risk R1).  Flags
// are identical to the K=2 KNOWNBUG dual -- only the K dimension differs,
// isolating exactly the nested-of-struct shape that aborts.
contract C {
    struct S {
        uint256 a;
        uint256 b;
    }

    mapping(uint256 => S) m;

    function check(uint256 i, uint256 v) public {
        m[i].a = v;
        assert(m[i].a == v);
    }
}
