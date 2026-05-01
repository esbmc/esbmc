// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.19;

// Test user-defined value types with wrap/unwrap.
type MyUint is uint16;

contract UDVTest {
    function test() public pure {
        // wrap: uint16 → MyUint
        MyUint a = MyUint.wrap(100);
        MyUint b = MyUint.wrap(200);

        // unwrap: MyUint → uint16
        uint16 rawA = MyUint.unwrap(a);
        uint16 rawB = MyUint.unwrap(b);

        assert(rawA == 100);
        assert(rawB == 200);
        assert(rawA + rawB == 300);
    }
}
