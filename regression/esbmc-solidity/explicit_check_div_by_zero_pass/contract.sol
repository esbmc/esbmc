pragma solidity >=0.8.0;

contract H {
    function check(uint256 a, uint256 b) public pure returns (uint256) {
        unchecked { return a / b; }
    }
}
