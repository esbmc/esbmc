// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bug B regression-lock: `delete fixedArr` for `uint[N]` must zero each
// element through the heap pointer, not NULL the pointer itself.
//
// Currently emits `ASSIGN this->a=0` (gen_zero(pointer) → NULL), so
// post-delete `a[0]` derefs a null pointer and reads nondet. KNOWNBUG.
contract C {
    uint[3] a;

    function f() public {
        require(a[0] == 0 && a[1] == 0 && a[2] == 0);
        a[0] = 10;
        a[1] = 20;
        a[2] = 30;
        delete a;
        assert(a[0] == 0);
        assert(a[1] == 0);
        assert(a[2] == 0);
    }
}
