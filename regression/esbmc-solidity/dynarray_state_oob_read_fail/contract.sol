// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Reading past the length of an empty dynamic-array state variable must be
// caught as an array-bounds violation under --bounds-check. The element buffer
// of a `T[]` state var is modelled as an infinity-sized array, so the explicit
// `index < <arr>_dynarray_len` claim in get_index_access_expr is what detects
// this out-of-bounds read (regression pin for the Solidity dyn-array OOB fix).
contract Base {
    uint[] test2;
    mapping(int => uint) test;
    constructor() {
        assert(test2[1] == 0);   // OUT OF BOUNDS: test2 is empty
        assert(test[1] == 0);    // mapping: never OOB
    }
}
