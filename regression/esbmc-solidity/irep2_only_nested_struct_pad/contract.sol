// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

// A struct declared inside a contract gets a tag qualified by the contract, so
// resolving the padded layout by the type's own unqualified name missed it and
// the literal kept fewer operands than its type describes.
contract C {
    struct Nested {
        string a;
        uint n;
    }
    Nested s;

    function f() public view returns (uint) {
        return s.n;
    }

    function g() public view {
        assert(f() == 0);
    }
}
