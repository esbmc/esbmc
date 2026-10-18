// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;
interface ITok { function balanceOf(address a) external view returns (uint256); }
abstract contract BaseE {
    uint32 public immutable RESCUE;
    ITok public immutable TOK;
    address public immutable FACTORY = msg.sender;
    constructor(uint32 r, ITok t) { RESCUE = r; TOK = t; }
}
abstract contract MidAbstract is BaseE {}
contract Leaf is MidAbstract {
    uint256 public s;
    constructor(uint32 r, ITok t) BaseE(r, t) {}
    function f(uint256 y) external { if (y > RESCUE) s = y; else s = 0; }
}
