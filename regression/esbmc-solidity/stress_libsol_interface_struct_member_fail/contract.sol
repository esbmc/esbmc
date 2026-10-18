// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Dual-to pass: the interface-hosted struct member access must succeed
// all the way to symex so that the negative assertion actually fires.

interface IThing {
    struct Info {
        address owner;
        uint256 value;
    }
}

contract C {
    function go() external pure {
        IThing.Info memory i = IThing.Info(address(0), 42);
        assert(i.value == 43); // deliberately wrong
    }
}
