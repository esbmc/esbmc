// Bug reachable within a single transaction: FAILED stays sound under bounding.
pragma solidity >=0.8.0;
contract C {
    function f(uint256 x) public pure { assert(x < 100); }
}
