// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Fixed-bytes rendering: a bytes4 param whose value a branch depends on, plus a
// bytes4 constructor arg. The generated test must use bytes4(..) literals of the
// exact declared width (regression for the recovered-value width defect and the
// constructor bytesN default), never bytes32(..).
contract KB {
    bytes4 public tag;
    bool public hit;
    constructor(bytes4 seed) { tag = seed; }
    function poke(bytes4 x) external { if (x == bytes4(0x12345678)) hit = true; }
}
