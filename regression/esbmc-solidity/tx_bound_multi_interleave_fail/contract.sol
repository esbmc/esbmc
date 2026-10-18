// Bug requires interleaving TWO different functions across two transactions:
// setA() in tx1, trigger() in tx2. Shows the bounded dispatcher preserves
// per-tx function-choice nondeterminism (not collapsed to a single function).
pragma solidity >=0.8.0;
contract C {
    bool a; bool bug;
    function setA() public { a = true; }
    function trigger() public { if (a) bug = true; assert(!bug); }
}
