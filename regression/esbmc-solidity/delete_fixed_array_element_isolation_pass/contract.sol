// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Solidity spec: `delete a[i]` resets only element `i`, leaves all
// others (and the array length) untouched.
//
// CORE: locks element-level delete behaviour so the upcoming
// emit_delete_block refactor doesn't regress it. Element write is
// already correct because the lhs is `a[i]`, not `a` itself.
contract C {
    uint[3] a;

    function f() public {
        require(a[0] == 0 && a[1] == 0 && a[2] == 0);
        a[0] = 10;
        a[1] = 20;
        a[2] = 30;
        delete a[1];
        assert(a[0] == 10);
        assert(a[1] == 0);
        assert(a[2] == 30);
    }
}
