// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.5.0;

contract Test {
    // function t(uint x, uint y, uint k) public returns (uint) {
    //     return (x + y) % k;
    // }
    function test() public {
        uint xx = 1;
        uint256 x = addmod(xx, 2, 3);
        assert(x == 0);
    }
}
