// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;

contract Base {
    uint data;
    constructor() {
        data = 1;
    }
    /// @custom:scribble if_succeeds {:msg "P0"} y == x + 1;
    // function test() external virtual {
    //     assert(data == 3);
    // }
}
