// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// T2.1 Stage S0 — KNOWNBUG pinning multi-arg conflation in abi.encodePacked.
// Same bug as abi_encode_multi_arg_distinct: lowering takes only the first
// compatible arg. Will flip to CORE at Stage S1.
contract H {
    function check(uint256 a, uint256 b, uint256 c) external pure {
        require(b != c);
        assert(keccak256(abi.encodePacked(a, b)) != keccak256(abi.encodePacked(a, c)));
    }
}
