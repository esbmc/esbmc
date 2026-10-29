// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: `mapping(K => T[]).push(x)` across a contract instantiated
// via `new C()` from another contract used to crash the Solidity
// frontend with `std::out_of_range: basic_string::substr: __pos (which
// is 4) > this->size() (which is 0)`.
//
// Root cause: `get_new_mapping_index_access` routed any non-scalar
// mapping value through the struct-shaped helper path which calls
// `get_mapping_struct_function`, whose `substr(prefix.length())` with
// prefix="tag-" (len 4) throws on the empty identifier of a
// pointer-typed dynamic-array value.  See
// `src/solidity-frontend/solidity_convert_mapping.cpp`.

contract C {
    mapping(address => uint256[]) m;
    function pushOne(address a, uint256 x) public { m[a].push(x); }
}

contract T {
    function test() public {
        // Construct C from another contract — this is what activates
        // the `is_new_expr` code path in the frontend and used to
        // crash on the mapping-of-dynamic-array state variable.
        new C();
    }
}
