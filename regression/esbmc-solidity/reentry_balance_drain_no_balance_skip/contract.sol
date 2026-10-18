pragma solidity >=0.8.0;
contract C {
    uint256 x;
    function set(uint256 v) external { x = v; }
    function get() external view returns (uint256) { return x; }
}
