// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Regression for find_decl_ref skipping interface bodies. Before the fix,
// `info.value` (where Info is a struct defined inside the interface)
// aborted with "cannot find struct member reference" during Converting.

interface IThing {
    struct Info {
        address owner;
        uint256 value;
    }
}

contract C {
    function readValue(IThing.Info calldata info) external pure returns (uint256) {
        return info.value;
    }

    function go() external pure {
        IThing.Info memory i = IThing.Info(address(0), 42);
        assert(i.value == 42);
    }
}
