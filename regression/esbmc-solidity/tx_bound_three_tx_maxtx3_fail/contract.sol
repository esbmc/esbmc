// Raising the bound to 3 transactions recovers the deep bug missed at N=2.
pragma solidity >=0.8.0;
contract C {
    uint256 count;
    function step() public { count += 1; assert(count < 3); }
}
