// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.0;

// Adversarial test: postfix ++/-- must return the OLD value and mutate afterward.
// If postfix were wrongly treated as prefix, multiple asserts would fail.
contract PostfixTest {
    function test_postfix() public pure {
        // Basic: capture old value via assignment
        uint x = 7;
        uint old_x = x++;          // old_x == 7, x == 8
        assert(old_x == 7);
        assert(x == 8);

        uint y = 3;
        uint old_y = y--;          // old_y == 3, y == 2
        assert(old_y == 3);
        assert(y == 2);

        // Chain: accumulate using postfix return value
        // x starts at 8
        uint sum = x++;            // sum = 8, x = 9
        sum += x++;                // sum = 8+9 = 17, x = 10
        sum += x++;                // sum = 17+10 = 27, x = 11
        assert(sum == 27);
        assert(x == 11);

        // Mixed prefix/postfix: the key distinguisher
        int a = 5;
        int b = a++;               // b = 5 (old), a = 6
        int c = ++a;               // a = 7, c = 7 (new)
        assert(b == 5);
        assert(c == 7);
        assert(a == 7);

        // Postfix in ternary condition
        uint p = 10;
        // p++ evaluates to 10, condition is true; after eval p == 11
        uint q = (p++ > 9) ? p : 0;
        assert(q == 11);

        // Snapshot pattern: each snap captures old value
        uint acc = 100;
        uint snap1 = acc--;       // snap1 = 100, acc = 99
        uint snap2 = acc--;       // snap2 = 99,  acc = 98
        uint snap3 = acc--;       // snap3 = 98,  acc = 97
        assert(snap1 + snap2 + snap3 == 297);
        assert(acc == 97);
    }
}
