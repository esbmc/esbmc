// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;

contract A {
    uint public counter;
    function inc() public virtual {
        counter += 1;
    }
}
contract B is A {
    function inc() public override {
        super.inc();
        counter += 10;
    }
}
contract C is B {
    // Wrong: asserts increment is 5, but super.inc() actually increments by 11
    function test() public {
        uint before = counter;
        super.inc();
        assert(counter == before + 5);
    }
}
