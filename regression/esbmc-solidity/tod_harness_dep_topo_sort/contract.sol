// SPDX-License-Identifier: MIT
// Regression for TOD harness dependency emission order.
//
// Source declares contracts in an order where a base appears AFTER a
// derived in the top-level node list (this can also happen when the
// ESBMC AST preprocessor reorders nodes).  Before the topological-sort
// fix, the harness emitter copied AST order verbatim, producing a
// harness file that solc rejects with "Definition of base has to
// precede definition of derived contract".
//
// Additionally, when --contract targets a BASE (e.g. Base below), any
// sibling contract that INHERITS from it must not be emitted into the
// harness at all — otherwise the derived contract's `is Base` reference
// appears before Base's declaration and compilation fails.
//
// After the fix:
//   - collect_dependency_definitions filters deps to only the target's
//     transitive bases (plus libraries / structs / enums, always kept).
//   - the kept set is topo-sorted so bases precede derived.
// Expected: the harness compiles and VERIFICATION SUCCESSFUL on the
// commutative add/inc pair.
pragma solidity >=0.8.0;

library L {
    function add(uint a, uint b) internal pure returns (uint) { return a + b; }
}

// Source-order trick: Derived appears BEFORE its own explicit `is Base`
// — fine for solc-standalone because Base is later in the file and solc
// uses global scoping, but if the harness emitter ever reorders these,
// the emitted file must still compile.
abstract contract IStep {
    function step() external virtual;
}

contract Base is IStep {
    using L for uint;
    uint public x;

    function step() external override {
        x = x.add(1);
    }

    function add(uint v) public {
        x = x.add(v);
    }

    function inc() public {
        x = x.add(1);
    }
}

// Not used by the harness — must be filtered out because including it
// would place `contract Derived is Base` BEFORE `contract Base` in some
// orderings.
contract Derived is Base {
    using L for uint;
    function extra() public { x = x.add(2); }
}
