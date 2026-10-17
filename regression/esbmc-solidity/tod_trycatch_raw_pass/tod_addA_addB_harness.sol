// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Counter
// Pairs (1):
//   - addA vs addB
//
// Verify each TOD_<a>_<b> contract with:
//   esbmc <this-file>.sol --contract TOD_<a>_<b> --bound --no-standard-checks --unwind 2 --no-unwinding-assertions
// Or let ESBMC drive all pairs via --tod-auto.

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// ===== Copy 1 =====
contract Counter_C1 {
    uint public counter;

    constructor() {
        counter = 0;
    }

    function addA() public {
        counter = counter + 1;
    }

    function addB() public {
        counter = counter + 1;
    }
}

// ===== Copy 2 =====
contract Counter_C2 {
    uint public counter;

    constructor() {
        counter = 0;
    }

    function addA() public {
        counter = counter + 1;
    }

    function addB() public {
        counter = counter + 1;
    }
}

// ===== TOD Harness contracts =====
// ----- addA vs addB -----
// Targeted state variables (referenced by BOTH functions):
//   - counter
contract TOD_addA_addB {
    function test(
    ) public {
        Counter_C1 c1 = new Counter_C1();
        Counter_C2 c2 = new Counter_C2();

        // Order 1: addA then addB
        c1.addA();
        c1.addB();

        // Order 2: addB then addA
        c2.addB();
        c2.addA();

        // State comparison — if any assert fails, TOD exists
        assert(c1.counter() == c2.counter());
    }
}

