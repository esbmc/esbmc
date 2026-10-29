// Multiple public functions; the bounded dispatcher still gives full per-tx
// function-choice nondeterminism. Invariant total <= 2*CAP holds across any
// sequence (require gates keep it inductive); k-induction converges.
pragma solidity >=0.8.0;
contract C {
    uint256 constant CAP = 1000000;
    uint256 total;
    function add(uint256 v) public {
        require(v <= CAP);
        require(total <= CAP);
        total += v;
    }
    function check() public view { assert(total <= 2 * CAP); }
}
