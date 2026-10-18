// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

interface IReader {
    function label() external view returns (string memory);
    function amount() external view returns (uint256);
}
interface IWriter {
    function poke(address who) external returns (bool);
}

// Vault inherits both interfaces so they materialize in the symbol table (the
// same shape as FarmingPool `is ERC20 is IERC20Metadata`), and its constructor
// takes two interface handles that must be distinct + nonzero — reconstructing
// this deploy requires synthesizing an ESBMCMock_* for each interface.
contract Vault is IReader, IWriter {
    IReader public r;
    IWriter public w;
    uint256 public s;

    constructor(IReader r_, IWriter w_) {
        require(address(r_) != address(0), "zero-reader");
        require(address(w_) != address(0), "zero-writer");
        require(address(r_) != address(w_), "same");
        r = r_;
        w = w_;
    }

    function label() external pure returns (string memory) { return "v"; }
    function amount() external pure returns (uint256) { return 3; }
    function poke(address) external pure returns (bool) { return true; }

    function step(uint256 x) external {
        if (x > 10) { s = x; } else { s = 1; }
    }
}
