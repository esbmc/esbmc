// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.0;
// A void-returning call can be observed (the return-value wrapper form could
// not express this — see docs/claude/solidity/revert-observation.md).
contract Sink {
    uint256 public s;
    function poke(uint256 x) external { require(x != 0, "zero"); s = x; }
}
contract Harness {
    Sink sink;
    function __ESBMC_reverted() internal returns (bool) {}
    function __ESBMC_assume(bool) internal pure {}
    constructor() { sink = new Sink(); }
    function check(uint256 x) public {
        __ESBMC_assume(x == 0);
        sink.poke(x);                 // void call, reverts on x == 0
        assert(__ESBMC_reverted());
    }
}
