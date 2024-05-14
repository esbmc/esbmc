// // SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    struct Book {
        string title;
        string author;
        uint book_id;
    }

    function setBook() public {
        Book memory book = Book("Learn Java", "TP", 1);
        assert(book.book_id + book.book_id == 2);
    }
}