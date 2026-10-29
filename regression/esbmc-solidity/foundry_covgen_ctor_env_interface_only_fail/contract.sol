// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

interface IFeed {
    function rate() external view returns (uint256);
}

// The constructor's ONLY parameter is an interface handle (never recovered as a
// scalar, so ctor_args is empty) AND it reads block.timestamp — this reaches the
// deploy-time env carrier. The carrier must route through build_call so the
// parameterized ctor renders as `new Gauge(<mock>)` (an ESBMCMock_IFeed), NOT an
// uncompilable bare `new Gauge()`.
contract Gauge is IFeed {
    IFeed public feed;
    uint256 public ts;
    uint256 public s;

    constructor(IFeed feed_) {
        require(address(feed_) != address(0), "zero");
        feed = feed_;
        ts = block.timestamp;
    }

    function rate() external pure returns (uint256) { return 1; }

    function tick(uint256 x) external {
        if (x > ts) { s = x; } else { s = 0; }
    }
}
