// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// Bug C: `delete o` for a struct containing a nested struct field
// currently CRASHES the symex (core dump). gen_zero recurses into the
// nested-struct field's symbol type but has no `symbol` case, returning
// nil. The IR emitted is `ASSIGN this->o={ .inn=nil, .top=0 }` — symex
// crashes on the nil operand.
//
// Regression-lock the expected post-delete state. KNOWNBUG until the
// recursive emit_delete_block resolves symbol types via ns and recurses
// per-field.
contract C {
    struct Inner {
        uint x;
        uint y;
    }
    struct Outer {
        Inner inn;
        uint top;
    }
    Outer o;

    function f() public {
        require(o.inn.x == 0 && o.inn.y == 0 && o.top == 0);
        o.inn.x = 1;
        o.inn.y = 2;
        o.top = 3;
        delete o;
        assert(o.inn.x == 0);
        assert(o.inn.y == 0);
        assert(o.top == 0);
    }
}
