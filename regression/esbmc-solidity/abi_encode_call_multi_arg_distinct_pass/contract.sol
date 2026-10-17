// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.1 Stage S0 — KNOWNBUG pinning multi-arg conflation in abi.encodeCall.
// Will flip at S1 once the multi-arg fold lands.
contract H {
    function f(uint256, uint256) external pure {}

    function check(uint256 b, uint256 c1, uint256 c2) external view {
        require(c1 != c2);
        bytes32 h1 = keccak256(abi.encodeCall(this.f, (b, c1)));
        bytes32 h2 = keccak256(abi.encodeCall(this.f, (b, c2)));
        assert(h1 != h2);
    }
}
