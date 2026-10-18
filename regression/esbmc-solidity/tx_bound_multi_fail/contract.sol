// Two public functions; either reaches a violating input within one tx.
pragma solidity >=0.8.0;
contract C {
    function f(uint256 a) public pure { assert(a != 42); }
    function g(uint256 b) public pure { assert(b != 7); }
}
