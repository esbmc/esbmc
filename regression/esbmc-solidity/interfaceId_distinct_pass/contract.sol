// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Ledger #15: distinct interfaces should have distinct interfaceIds.
// Real EVM: derived from selector XOR — collision-free in practice.
// Pre-fix model: both nondet, SMT can pick them equal.
interface IA {
    function aa() external;
}
interface IB {
    function bb() external;
}

contract Test {
    function check() public pure {
        bytes4 a = type(IA).interfaceId;
        bytes4 b = type(IB).interfaceId;
        assert(a != b);
    }
}
