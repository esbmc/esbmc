// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Regression: calling the auto-generated public getter on a nested
// mapping (`a.m(x, y)`) used to crash the Solidity frontend with
// std::out_of_range from substr(4) — the value type was MAPPING and
// the struct-shaped path stripped a "tag-" prefix from an empty
// identifier.  Now: the call must lower without crashing.
contract A {
    mapping(address => mapping(address => uint256)) public m;
}

contract Test {
    function check(address x, address y) public {
        A a = new A();
        uint256 r = a.m(x, y);
        assert(r == r); // trivial; we only care that the call compiles
    }
}
