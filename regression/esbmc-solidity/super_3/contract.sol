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
    // super.inc() calls B.inc, which calls A.inc (+1) then adds 10
    // So each call to super.inc() increases counter by exactly 11
    function test() public {
        uint before = counter;
        super.inc();
        assert(counter == before + 11);
    }
}
