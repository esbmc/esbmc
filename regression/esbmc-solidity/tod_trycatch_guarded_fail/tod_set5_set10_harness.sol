// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Counter
// Pairs (1):
//   - set5 vs set10
//
// Verify each TOD_<a>_<b> contract with:
//   esbmc <this-file>.sol --contract TOD_<a>_<b> --bound --no-standard-checks --unwind 2 --no-unwinding-assertions
// Or let ESBMC drive all pairs via --tod-auto.

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// ===== Copy 1 =====
contract Counter_C1 {
    uint public x;

    constructor() {
        x = 0;
    }

    function set5() public {
        require(x == 0);
        x = 5;
    }

    function set10() public {
        require(x == 0);
        x = 10;
    }
}

// ===== Copy 2 =====
contract Counter_C2 {
    uint public x;

    constructor() {
        x = 0;
    }

    function set5() public {
        require(x == 0);
        x = 5;
    }

    function set10() public {
        require(x == 0);
        x = 10;
    }
}

// ===== TOD Harness contracts =====
// ----- set5 vs set10 -----
// Targeted state variables (referenced by BOTH functions):
//   - x
contract TOD_set5_set10 {
    function test(
    ) public {
        Counter_C1 c1 = new Counter_C1();
        Counter_C2 c2 = new Counter_C2();

        // Order 1: set5 then set10
        try c1.set5() {} catch {}
        try c1.set10() {} catch {}

        // Order 2: set10 then set5
        try c2.set10() {} catch {}
        try c2.set5() {} catch {}

        // State comparison — if any assert fails, TOD exists
        assert(c1.x() == c2.x());
    }
}

