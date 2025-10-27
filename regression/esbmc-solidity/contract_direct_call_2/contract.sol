// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract A {
    uint public count;

    function increment() public virtual {
        count += 1;
        assert(count == 1);  
    }
}

contract B is A {
    function increment() public override {
        A.increment(); 
    }
}
