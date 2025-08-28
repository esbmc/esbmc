// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Base {
    function call(uint a) public pure returns (uint) {
        return a * 2;
    }
}

contract ExampleFail {
    Base b = new Base();
    uint public x = b.call(1);
}
