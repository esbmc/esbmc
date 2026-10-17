// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Stress: state-var 2D dynamic array whose ELEMENT TYPE is a struct
// `S { uint256 amount; address holder; }`. Exercises push of a
// struct-by-value into a nested dyn-array slot, with field-level
// invariant checks after each operation.
contract C {
    struct S {
        uint256 amount;
        address holder;
    }

    S[][] internal book;

    function pushRow() internal {
        book.push();
    }

    function pushS(uint256 r, uint256 amt, address h) internal {
        S memory s;
        s.amount = amt;
        s.holder = h;
        book[r].push(s);
    }

    function run() external {
        address h1 = address(0x1);
        address h2 = address(0x2);
        address h3 = address(0x3);

        assert(book.length == 0);

        pushRow();
        pushRow();
        assert(book.length == 2);
        assert(book[0].length == 0);
        assert(book[1].length == 0);

        pushS(0, 100, h1);
        pushS(0, 200, h2);
        pushS(0, 300, h3);
        assert(book[0].length == 3);
        assert(book[0][0].amount == 100);
        assert(book[0][0].holder == h1);
        assert(book[0][1].amount == 200);
        assert(book[0][1].holder == h2);
        assert(book[0][2].amount == 300);
        assert(book[0][2].holder == h3);

        pushS(1, 999, h1);
        assert(book[1].length == 1);
        assert(book[1][0].amount == 999);
        assert(book[1][0].holder == h1);

        // mutate field of an existing slot
        book[0][1].amount = 250;
        assert(book[0][1].amount == 250);
        assert(book[0][1].holder == h2);

        // pop and re-push
        book[0].pop();
        assert(book[0].length == 2);
        assert(book[0][1].amount == 250);

        pushS(0, 777, h3);
        assert(book[0].length == 3);
        assert(book[0][2].amount == 777);
        assert(book[0][2].holder == h3);

        // outer pop empty row
        pushRow();
        assert(book.length == 3);
        book.pop();
        assert(book.length == 2);
        assert(book[1][0].amount == 999);
        assert(book[0][0].amount == 100);
    }
}
