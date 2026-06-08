// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract StructNested {
    struct Inner {
        uint256 x;
        bool flag;
    }

    struct Outer {
        Inner data;
        uint256 id;
        address owner;
    }

    function test_nested() public pure {
        Inner memory inner = Inner(42, true);
        assert(inner.x == 42);
        assert(inner.flag == true);

        Outer memory outer = Outer(inner, 1, address(0));
        assert(outer.data.x == 42);
        assert(outer.data.flag == true);
        assert(outer.id == 1);

        // Modify nested field
        outer.data.x = 100;
        assert(outer.data.x == 100);
        // Original inner should be separate (memory copy semantics)
        // Note: in Solidity memory, structs are copied, not referenced
    }

    function test_struct_default() public pure {
        Inner memory i;
        assert(i.x == 0);
        assert(i.flag == false);
    }
}
