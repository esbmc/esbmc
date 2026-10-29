// Auto-generated TOD (Transaction Order Dependence) harness
// Contract: Bal
// Pair:     payFixed vs payIfRich
// Mode:     balance

// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

// TOD classification helpers.  An assertion failure inside one
// of these functions tells the user which TOD category fired.
function __tod_race_check(bool cond) pure {
    assert(cond); // TOD-Race: non-commutative state update
}
function __tod_balance_check(bool cond) pure {
    assert(cond); // TOD-Balance: order-dependent ETH movement
}

// ===== Copy 1 =====
contract Bal_C1 {
    constructor() payable {}

    function payFixed(address payable to) public {
        if (to == address(this)) return; // skip self-transfer
        to.transfer(50);
    }

    function payIfRich(address payable to) public {
        if (to == address(this)) return; // skip self-transfer
        if (address(this).balance >= 150) {
            to.transfer(100);
        }
    }

    function __tod_bal() public view returns (uint) { return address(this).balance; }
    function __tod_addr() public view returns (address) { return address(this); }
}

// ===== Copy 2 =====
contract Bal_C2 {
    constructor() payable {}

    function payFixed(address payable to) public {
        if (to == address(this)) return; // skip self-transfer
        to.transfer(50);
    }

    function payIfRich(address payable to) public {
        if (to == address(this)) return; // skip self-transfer
        if (address(this).balance >= 150) {
            to.transfer(100);
        }
    }

    function __tod_bal() public view returns (uint) { return address(this).balance; }
    function __tod_addr() public view returns (address) { return address(this); }
}

// ===== TOD harness =====
// ----- payFixed vs payIfRich -----
// Targeted state variables (referenced by BOTH functions): <none>
// Plus: address(this).balance (TOD-Balance check)
contract TOD_payFixed_payIfRich {
    function test(
        uint __initBal,
        address payable a_to,
        address payable b_to
    ) public payable {
        Bal_C1 c1 = new Bal_C1{value: __initBal}();
        Bal_C2 c2 = new Bal_C2{value: __initBal}();

        address __c1_addr = c1.__tod_addr();
        address __c2_addr = c2.__tod_addr();
        require(a_to != __c1_addr && a_to != __c2_addr, "isolate copies");
        require(b_to != __c1_addr && b_to != __c2_addr, "isolate copies");

        // Order 1: payFixed then payIfRich
        c1.payFixed(a_to);
        c1.payIfRich(b_to);

        // Order 2: payIfRich then payFixed
        c2.payIfRich(b_to);
        c2.payFixed(a_to);

        // State comparison — if any assert fails, TOD exists
        assert(c1.__tod_bal() == c2.__tod_bal());
    }
}

