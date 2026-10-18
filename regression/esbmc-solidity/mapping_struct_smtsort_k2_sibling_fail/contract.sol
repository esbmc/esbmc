// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// CORE FAILED soundness companion of mapping_struct_smtsort_k2_pass.
// Writes only struct field .a, asserts the untouched sibling .b equals
// the written value.  .b is independent (modelling-only nondet init),
// so the assertion must be violable: proves the per-field
// struct-of-arrays decomposition does NOT bleed a write of one field
// into a sibling field.
contract C {
    struct S {
        uint256 a;
        uint256 b;
    }

    mapping(uint256 => mapping(uint256 => S)) m;

    function check(uint256 i, uint256 j, uint256 v) public {
        m[i][j].a = v;
        assert(m[i][j].b == v);
    }
}
