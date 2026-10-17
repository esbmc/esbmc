// --solidity-precise restores the unbounded while(nondet) harness. The
// invariant is inductive (require pins msg.sender == owner each tx), so
// k-induction proves it for ANY number of transactions (a true unbounded proof).
pragma solidity >=0.8.0;
contract C {
    address owner;
    constructor() { owner = msg.sender; }
    function f() public view { require(msg.sender == owner); assert(msg.sender == owner); }
}
