// // SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    struct tt {
        Book a;
        uint b;
    }
    Book test;

    function setBook() public {
        test = Book("Learn Java", "TP", 1);
        Book memory book = Book("Learn Java", "TP", 1);
        tt memory x = tt(book, 1);
        assert(book.book_id + x.b == 2);
    }
}

struct Book {
    string title;
    string author;
    uint book_id;
}
