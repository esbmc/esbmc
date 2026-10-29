// State accumulates across transactions: count reaches 2 only with TWO calls.
// The default bound N=2 catches it (proving the harness explores >1 tx).
pragma solidity >=0.8.0;
contract C {
    uint256 count;
    function step() public { count += 1; assert(count < 2); }
}
