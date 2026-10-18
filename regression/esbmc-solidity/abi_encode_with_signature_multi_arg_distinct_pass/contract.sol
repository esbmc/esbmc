// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.1 Stage S0 — KNOWNBUG pinning multi-arg conflation in
// abi.encodeWithSignature. Same root cause; will flip at S1.
contract H {
    function check(uint256 b, uint256 c1, uint256 c2) external pure {
        require(c1 != c2);
        bytes32 h1 = keccak256(abi.encodeWithSignature("f(uint256,uint256)", b, c1));
        bytes32 h2 = keccak256(abi.encodeWithSignature("f(uint256,uint256)", b, c2));
        assert(h1 != h2);
    }
}
