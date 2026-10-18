// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// Revert fidelity: the `v == 7` branch reverts with a custom error. The Foundry
// coverage generator must wrap the covering call in vm.expectRevert() so the
// assertion-free replay stays a PASS in forge instead of aborting on the revert.
contract Rev {
    uint256 public x;
    error Denied(uint256 v);

    function poke(uint256 v) external {
        if (v == 7) revert Denied(v);
        x = v;
    }
}
