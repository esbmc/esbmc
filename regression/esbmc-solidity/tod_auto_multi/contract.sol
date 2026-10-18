// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Two independent TOD pairs.  --tod=auto should discover both
// (addX,mulX share x; doubleY,setY share y) and verify both as TOD.
contract MultiTod {
    uint public x;
    uint public y;

    constructor() { x = 1; y = 1; }

    function addX(uint n) public { x = x + n; }
    function mulX(uint n) public { x = x * n; }

    function setY(uint v) public { y = v; }
    function doubleY() public { y = y * 2; }
}
