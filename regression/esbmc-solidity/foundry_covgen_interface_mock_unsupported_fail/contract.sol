// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

struct Info {
    uint256 a;
    uint256 b;
}

interface IBox {
    function get() external view returns (Info memory);
}

// Holder inherits IBox so it materializes. IBox has a STRUCT return, which the
// generator cannot render as a default — so the mock is all-or-nothing rejected
// and the IBox constructor argument degrades to UNSUPPORTED (never a partial,
// uncompilable mock).
contract Holder is IBox {
    IBox public box;
    uint256 public s;

    constructor(IBox box_) {
        require(address(box_) != address(0), "zero");
        box = box_;
    }

    function get() external pure returns (Info memory) { return Info(1, 2); }

    function tick(uint256 x) external {
        if (x > 5) { s = x; } else { s = 0; }
    }
}
