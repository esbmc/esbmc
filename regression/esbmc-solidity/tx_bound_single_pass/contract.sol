// k-induction converges under the bounded-by-default harness (N=2).
// Single public function; assertion holds for any transaction count.
pragma solidity >=0.8.0;
contract C {
    uint256 x;
    function set(uint256 v) public { x = v; assert(x == v); }
}
