// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

// Return array of uint8 type
contract A {

    function int8_call() public virtual returns (uint8[2] memory) {
        uint8[2] memory a;
        a[0] = 21;
        a[1] = 42;
        return a;
    }
}

contract call {
    function test_return() public {
        A a = new A();
        assert(a.int8_call()[0] == 21);
        assert(a.int8_call()[1] == 42);
    }
}