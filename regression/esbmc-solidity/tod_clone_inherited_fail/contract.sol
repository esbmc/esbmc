// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// TOD on an inherited contract: tests that _ESBMC_clone_Derived
// correctly shallow-copies fields contributed by both Base and
// Derived.  Before the struct-copy refactor, clone silently no-op'd
// on inherited contracts because per-field emission missed the
// merged struct symbol, so cloned instances stayed at ctor defaults
// and no race was ever detected.
//
// Race-candidate functions are both defined on Derived so the
// harness pair lookup (which doesn't follow inheritance edges)
// finds them; the *fields* they mutate straddle Base and Derived,
// exercising the merged-struct clone path.
contract Base {
    uint public baseX;

    constructor() {
        baseX = 5;
    }
}

contract Derived is Base {
    uint public derivedY;

    constructor() {
        derivedY = 0;
    }

    // Reads inherited baseX, writes derivedY.
    function takeSnapshot() public {
        derivedY = baseX + derivedY;
    }

    // Writes inherited baseX.
    function scaleBase(uint n) public {
        baseX = baseX * n;
    }
}
