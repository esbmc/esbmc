// Bug needs THREE transactions (count reaches 3). Default bound N=2 misses it,
// yielding a bounded SUCCESSFUL. Pinned KNOWNBUG to document that bounded-mode
// SUCCESSFUL is an under-approximation, not an unbounded proof. The companion
// test tx_bound_three_tx_maxtx3_fail recovers the bug with --solidity-max-tx 3.
pragma solidity >=0.8.0;
contract C {
    uint256 count;
    function step() public { count += 1; assert(count < 3); }
}
