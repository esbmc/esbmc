// --solidity-max-tx 1 elides the dispatcher loop to a single transaction.
// The 2-tx accumulation bug is (deliberately) not reachable in one tx.
pragma solidity >=0.8.0;
contract C {
    uint256 count;
    function step() public { count += 1; assert(count < 2); }
}
