// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Base {
    uint public data;
    constructor() {
        data = 1;
    }
    /// @custom:scribble if_succeeds {:msg "P0"} y == x + 1;
    // function add() external {
    //     data += 1;
    // }
}

contract Derived is Base {
    constructor(uint x) Base() {
        data = x;
    }

    // The overriding function may only change the visibility of the overridden function from external to public
}

contract Derived2 is Derived {
    constructor()  Derived(1) {
       assert(data == 1);
    }

}
