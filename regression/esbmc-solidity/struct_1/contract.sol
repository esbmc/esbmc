// // SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Base {
    struct Book {
        string title;
        string author;
        uint book_id;
    }
    Book book;

    function setBook() public {
        book = Book("Learn Java", "TP", 1);
        assert(getBook() == 1);
    }

    function getBook() public returns (uint) {
        return book.book_id;
    }
}