// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bug B nested form: a struct containing a `uint[N]` field. The
// recursive emit_delete_block must descend into the struct, identify
// the fixed-array field via #sol_array_size, and element-zero through
// its heap pointer.
//
// Currently the gen_zero approach NULLs the inner pointer. KNOWNBUG.
contract C {
    struct S {
        uint x;
        uint[3] a;
    }
    S s;

    function f() public {
        require(s.x == 0 && s.a[0] == 0 && s.a[1] == 0 && s.a[2] == 0);
        s.x = 99;
        s.a[0] = 1;
        s.a[1] = 2;
        s.a[2] = 3;
        delete s;
        assert(s.x == 0);
        assert(s.a[0] == 0);
        assert(s.a[1] == 0);
        assert(s.a[2] == 0);
    }
}
